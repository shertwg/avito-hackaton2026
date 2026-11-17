import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl

from src.common.itemstats import build_features,feature_names
from src.common.preparation import write_json,fingerprint
from src.common.similarity import attach_features
from scripts.audit_itemstats_features import audit


class ItemstatsTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.prepared=self.root/'prepared'
        source={'complete':True,'splits':{}}
        entries={
            'train':[[ (10,1,0,0),(10,1,1,1),(20,2,2,1),(99,9,3,0)],
                     [(10,1,0,1),(10,1,1,0),(20,2,1,1)],
                     [(10,1,0,0),(30,2,1,0),(20,2,0,1)],
                     [(30,2,0,1),(40,7,1,1)]],
            'val':[[(10,1,0,1),(10,1,1,0),(99,9,2,0),(40,7,3,1)]],
            'test':[[(40,7,0,0),(20,2,1,0)]]}
        for split,parts in entries.items():
            saved=[]
            for index,rows in enumerate(parts):
                frame=pl.DataFrame({'split':[split]*len(rows),'feed_id':[index]*len(rows),
                    'candidate_index':list(range(len(rows))),'item_id':[r[0] for r in rows],
                    'category_id':[r[1] for r in rows],'pos':[r[2] for r in rows],'target':[r[3] for r in rows]},
                    schema_overrides={'feed_id':pl.Int32,'candidate_index':pl.UInt32,'item_id':pl.Int32,
                                      'category_id':pl.Int8,'pos':pl.Int16,'target':pl.Int8})
                if split=='test':
                    frame=frame.drop('target','pos')
                name=f'part-{index:05d}.parquet'
                relative=Path(split)/'candidates'/name
                (self.prepared/relative).parent.mkdir(parents=True,exist_ok=True)
                frame.write_parquet(self.prepared/relative)
                items,categories,events=([10,10,99],[1,1,9],[1,2,10]) if index==1 else ([10,20],[1,2],[2,1])
                feeds=pl.DataFrame({'feed_id':[index],'history__item_id':[items],
                    'history__category_id':[categories],'history__event_type':[events]})
                feed_relative=Path(split)/'feeds'/name
                (self.prepared/feed_relative).parent.mkdir(parents=True,exist_ok=True)
                feeds.write_parquet(self.prepared/feed_relative)
                saved.append({'source':name,'candidate_path':relative.as_posix(),'feed_path':feed_relative.as_posix(),
                              'candidates':len(rows),'feeds':1})
            source['splits'][split]={'parts':saved,'candidates':sum(p['candidates'] for p in saved)}
        write_json(self.prepared/'manifest.json',source)

    def build(self,name):
        return build_features(self.prepared,self.root/name,train_parts=2,memory_mb=256,threads=1)

    def test_formulas_exclusion_and_unseen_values(self):
        manifest=self.build('features')
        self.assertEqual(manifest['donor_summary']['impressions'],6)
        self.assertEqual(manifest['donor_summary']['clicks'],3)
        self.assertTrue(set(manifest['excluded_train_sources']).isdisjoint(manifest['identity']['donor_sources']))
        frame=pl.read_parquet(self.root/'features/train/part-00000.parquet')
        self.assertEqual(frame.height,4)
        self.assertEqual(len(feature_names()),32)
        row=frame.row(0,named=True)
        self.assertEqual(row['is_item_impressions'],3)
        self.assertEqual(row['is_item_clicks'],1)
        self.assertAlmostEqual(row['is_item_expected_clicks'],5/3,places=6)
        self.assertAlmostEqual(row['is_item_ctr'],11/23,places=6)
        self.assertAlmostEqual(row['is_item_position_ratio'],33/35,places=6)
        self.assertEqual(row['is_item_history_count'],3)
        self.assertEqual(row['is_item_history_evt1'],1)
        self.assertEqual(row['is_item_history_evt2'],2)
        unknown=frame.filter(pl.col('item_id')==99).row(0,named=True)
        self.assertEqual(unknown['is_item_impressions'],0)
        self.assertEqual(unknown['is_item_ctr'],.5)
        self.assertEqual(unknown['is_item_position_ratio'],1)
        self.assertEqual(unknown['is_item_history_evt10'],1)
        self.assertTrue(audit(self.prepared,self.root/'features',self.root/'audit.json')['complete'])
        path=self.root/'features/train/part-00000.parquet'
        frame.with_columns((pl.col('is_item_ctr')+.01).alias('is_item_ctr')).write_parquet(path)
        manifest['splits']['train']['parts'][path.name]['fingerprint']=fingerprint(path)
        write_json(self.root/'features/manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'independent itemstats formula mismatch'):
            audit(self.prepared,self.root/'features',self.root/'invalid-audit.json')
        frame.write_parquet(path)
        manifest['splits']['train']['parts'][path.name]['fingerprint']=fingerprint(path)
        write_json(self.root/'features/manifest.json',manifest)
        with self.assertRaisesRegex(ValueError,'overlap'):
            attach_features(frame,self.root/'features','train',['part-00001.parquet'])
        baseline={p.relative_to(self.root/'features'):pl.read_parquet(p) for p in (self.root/'features').glob('*/*.parquet')}
        source=json.loads((self.prepared/'manifest.json').read_text())
        for split,indices in [('train',[0,3]),('val',[0])]:
            for index in indices:
                path=self.prepared/source['splits'][split]['parts'][index]['candidate_path']
                pl.read_parquet(path).with_columns((1-pl.col('target')).alias('target')).write_parquet(path)
        self.build('changed_labels')
        for relative,expected in baseline.items():
            actual=pl.read_parquet(self.root/'changed_labels'/relative)
            np.testing.assert_array_equal(expected.select(feature_names()).to_numpy().view(np.uint32),
                                          actual.select(feature_names()).to_numpy().view(np.uint32))
        with self.assertRaisesRegex(ValueError,'configuration changed'):
            build_features(self.prepared,self.root/'features',train_parts=1,resume=True)

    def test_interrupted_aggregation_and_parquet_resume(self):
        def interrupt(path,value):
            write_json(path,value)
            if value.get('stage')=='item_exposure':
                raise RuntimeError('interrupted')
        with patch('src.common.itemstats.write_json',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                self.build('features')
        self.assertFalse(json.loads((self.root/'features/manifest.json').read_text())['complete'])
        manifest=build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        self.assertTrue(manifest['complete'])
        before={p:p.stat().st_mtime_ns for p in (self.root/'features').glob('*/*.parquet')}
        saved_bytes=(self.root/'features/manifest.json').read_bytes()
        build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        self.assertEqual((self.root/'features/manifest.json').read_bytes(),saved_bytes)
        manifest['complete']=False
        write_json(self.root/'features/manifest.json',manifest)
        def interrupt_features(path,value):
            write_json(path,value)
            if value.get('stage')=='features_train':
                raise RuntimeError('feature interruption')
        with patch('src.common.itemstats.write_json',side_effect=interrupt_features):
            with self.assertRaisesRegex(RuntimeError,'feature interruption'):
                build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        saved=json.loads((self.root/'features/manifest.json').read_text())
        self.assertEqual(saved['splits']['train']['covered_parts'],2)
        self.assertEqual(saved['splits']['val']['covered_parts'],1)
        build_features(self.prepared,self.root/'features',train_parts=2,memory_mb=256,threads=1,resume=True)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})


    def test_array_lookup_sql_equivalence_and_legacy_resume(self):
        import duckdb
        from src.common.itemstats import LEGACY_IMPLEMENTATIONS
        output = self.root / 'features'
        manifest = self.build('features')
        source = json.loads((self.prepared / 'manifest.json').read_text())
        connection = duckdb.connect(str(output / '_statistics.duckdb'), read_only=True)
        try:
            for split, info in manifest['splits'].items():
                entries = {entry['source']: entry for entry in source['splits'][split]['parts']}
                for name, part in info['parts'].items():
                    candidates = pl.read_parquet(self.prepared / entries[name]['candidate_path'])
                    connection.register('batch', candidates.to_arrow())
                    query = ('SELECT b.split,b.feed_id,b.candidate_index,b.item_id,' + ','.join(feature_names())
                             + ' FROM batch b LEFT JOIN item_lookup USING(item_id) LEFT JOIN category_lookup USING(category_id)'
                             + ' ORDER BY b.feed_id,b.candidate_index')
                    sql = pl.from_arrow(connection.execute(query).fetch_arrow_table())
                    actual = pl.read_parquet(output / part['path'])
                    self.assertTrue(actual.select('split','feed_id','candidate_index','item_id').equals(
                        sql.select('split','feed_id','candidate_index','item_id')))
                    np.testing.assert_array_equal(actual.select(feature_names()).to_numpy().view(np.uint32),
                                                  sql.select(feature_names()).to_numpy().view(np.uint32))
        finally:
            connection.close()
        self.assertEqual(len(manifest['artifacts']), 4)
        legacy_hash = next(iter(LEGACY_IMPLEMENTATIONS))
        manifest['identity']['implementation_sha256'] = legacy_hash
        for key in ('statistics_value_signature', 'lookup_implementation_sha256'):
            manifest['identity'].pop(key)
        manifest.pop('compatible_producer_implementations')
        manifest.pop('artifacts')
        for info in manifest['splits'].values():
            for part in info['parts'].values():
                part.pop('producer_implementation_sha256')
        write_json(output / 'manifest.json', manifest)
        saved = (output / 'manifest.json').read_bytes()
        build_features(self.prepared, output, train_parts=2, memory_mb=256, threads=1, resume=True)
        self.assertEqual((output / 'manifest.json').read_bytes(), saved)
        manifest['complete'] = False
        missing = output / 'train/part-00000.parquet'
        missing.unlink()
        write_json(output / 'manifest.json', manifest)
        repaired = build_features(self.prepared, output, train_parts=2, memory_mb=256, threads=1, resume=True)
        current_hash = repaired['identity']['implementation_sha256']
        self.assertIn(legacy_hash, repaired['compatible_producer_implementations'])
        self.assertIn(current_hash, repaired['compatible_producer_implementations'])
        self.assertEqual(repaired['splits']['train']['parts']['part-00000.parquet']['producer_implementation_sha256'], current_hash)
        self.assertEqual(repaired['splits']['train']['parts']['part-00003.parquet']['producer_implementation_sha256'], legacy_hash)
        self.assertTrue(audit(self.prepared, output, self.root/'repaired-audit.json')['complete'])


if __name__=='__main__':
    unittest.main()
