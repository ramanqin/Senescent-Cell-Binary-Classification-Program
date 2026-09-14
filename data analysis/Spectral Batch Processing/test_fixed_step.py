import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from spectral_preprocessor import PreprocessConfig, preprocess, save_spectrum, read_spectrum
from app import SpectrumPreprocessorApp

class Var:
    def __init__(self, value): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value

class FixedStepTests(unittest.TestCase):
    def test_default_and_old_constructor(self):
        for cfg in [PreprocessConfig(),PreprocessConfig(resample_step=3,resample_enabled=False)]:
            self.assertEqual(cfg.resample_step,1.0)
            self.assertIs(cfg.resample_enabled,True)
            self.assertEqual(cfg.sg_window,7)
            self.assertEqual(cfg.baseline_lambda,100000)

    def test_mutation_is_rejected(self):
        for key,value in [('resample_step',3),('resample_step',float('nan')),('resample_enabled',False)]:
            cfg=PreprocessConfig();setattr(cfg,key,value)
            with self.assertRaises(ValueError): cfg.validate()

    def test_old_json_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.json'
            for nested in [True,False]:
                values={'resample_step':3,'resample_enabled':False,'sg_window':19,'baseline_lambda':22222}
                p.write_text(json.dumps({'preprocess':values} if nested else values),encoding='utf-8')
                cfg=PreprocessConfig.load(p);cfg.save(p)
                saved=json.loads(p.read_text(encoding='utf-8'))
                self.assertEqual(saved['resample_step'],1)
                self.assertTrue(saved['resample_enabled'])
                self.assertEqual(saved['sg_window'],19)
                self.assertEqual(saved['baseline_lambda'],22222)

    def test_full_grid_and_output_round_trip(self):
        x=np.linspace(500,1900,481);y=10+np.sin(x/50)+np.exp(-((x-1000)/25)**2)
        result=preprocess(x,y,PreprocessConfig())
        np.testing.assert_array_equal(result.x,np.arange(600,1801))
        self.assertEqual(result.output_points,1201)
        self.assertTrue(np.isfinite(result.y).all())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.txt';save_spectrum(p,result.x,result.y)
            xx,yy=read_spectrum(p);np.testing.assert_array_equal(xx,result.x)
            np.testing.assert_allclose(yy,result.y,atol=1e-8)

    def test_multiple_ranges_and_no_extrapolation(self):
        x=np.arange(602,1799,2.7);y=10+np.sin(x/50)
        cfg=PreprocessConfig(ranges='600-800,1000-1100')
        r=preprocess(x,y,cfg)
        self.assertGreaterEqual(r.x.min(),x.min())
        self.assertTrue(np.all((r.x<=800)|(r.x>=1000)))
        self.assertEqual(r.x.max(),1100)
        self.assertTrue(np.all(np.diff(r.x)[np.diff(r.x)<100]==1))

    def test_gui_config_enforces_lock(self):
        from dataclasses import asdict
        variables={k:Var(v) for k,v in asdict(PreprocessConfig()).items()}
        variables['resample_step'].set(3);variables['resample_enabled'].set(False)
        obj=SimpleNamespace(vars=variables)
        cfg=SpectrumPreprocessorApp._config(obj)
        self.assertEqual(cfg.resample_step,1)
        self.assertTrue(variables['resample_enabled'].get())

    def test_gui_import_old_settings_announces_change(self):
        from dataclasses import asdict
        obj=SimpleNamespace(vars={k:Var(v) for k,v in asdict(PreprocessConfig()).items()},_write_log=lambda m:None)
        for k in ['extensions','recursive','preserve_tree','overwrite','output_format','output_suffix','precision']:setattr(obj,k,Var(None))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.json';p.write_text(json.dumps({'preprocess':{'resample_step':3,'resample_enabled':False,'sg_window':19}}))
            with patch('app.filedialog.askopenfilename',return_value=str(p)),patch('app.messagebox.showinfo') as info,patch('app.messagebox.showerror') as error:
                SpectrumPreprocessorApp._load_config(obj)
                error.assert_not_called();info.assert_called_once()
                self.assertEqual(obj.vars['resample_step'].get(),1)
                self.assertEqual(obj.vars['sg_window'].get(),19)

if __name__=='__main__':unittest.main()
