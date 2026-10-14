"""Integration and gradient checks against the real native materials."""
import unittest
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import numpy as np
import torch
import _fe2_material
from pyprnn import PRNN
from pyprnn.runtime import evaluate_with_tangent, observe

J2 = '''material = {type="J2"; rank=2; state="PlaneStress";
young=3130.; poisson=0.37; yield="64.80 - 33.60 * exp(x / -0.003407)";
rmTolerance=1.e-12; rmMaxIter=100;};'''
ELASTIC = 'material={type="Isotropic"; rank=2; state="PlaneStress"; young=3130.; poisson=0.37;};'


def network(properties=J2):
    model = PRNN(3,3,2,material_factory=_fe2_material.MaterialFactory(properties)).double()
    with torch.no_grad():
        model.fc1.weight.copy_(torch.cat([torch.eye(3), .7*torch.eye(3)]))
        model.fc1.weight[0,1] = .12
        model.fc2.weight.fill_(-3)
        for i in range(3):
            model.fc2.weight[i,i] = .4
            model.fc2.weight[i,3+i] = -.2
    return model


class BridgeTest(unittest.TestCase):
    def test_generic_factory_and_errors(self):
        types = _fe2_material.MaterialFactory.types()
        self.assertTrue({'J2','Isotropic','Melro','FE2'}.issubset(types))
        for config in [J2, ELASTIC]:
            m = _fe2_material.MaterialFactory(config)(2)
            with self.assertRaises(ValueError):
                m.update(np.zeros((2,4)))
            with self.assertRaises(ValueError):
                m.update(np.full((2,3),np.nan))
            stress, tangent = m.update(np.zeros((2,3)))
            self.assertEqual(tangent.shape,(2,3,3))
            np.testing.assert_array_equal(stress,0)
        with self.assertRaises(RuntimeError):
            _fe2_material.MaterialFactory('material={type="Missing"; rank=2;};')(1)

    def test_online_lifecycle_and_observations(self):
        m, reference = network(), network()
        a = np.array([.045,-.004,.01])
        b = np.array([.003,0,.001])
        sa,_ = evaluate_with_tangent(m,a)
        sb,_ = evaluate_with_tangent(m,a)
        np.testing.assert_allclose(sa,sb,rtol=0,atol=1.e-10)
        identity = m._online.native
        observe(m,b)
        self.assertIs(m._online.native, identity)

        evaluate_with_tangent(reference,a)
        m.commit(); reference.commit(); m.commit()
        expected,_ = evaluate_with_tangent(reference,b)
        actual,_ = evaluate_with_tangent(m,b)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1.e-9)
        evaluate_with_tangent(m,a*3)
        m.cancel(); m.commit()
        np.testing.assert_allclose(evaluate_with_tangent(m,b)[0],expected,atol=1.e-9)
        m.cancel()
        observe(m,a*4)
        m.commit()
        np.testing.assert_allclose(evaluate_with_tangent(m,b)[0],expected,atol=1.e-9)
        self.assertIs(m._online.native, identity)

    def test_native_calls_from_non_jem_python_workers(self):
        factory=_fe2_material.MaterialFactory(J2)
        strain=np.array([[.04,-.003,.01]])
        expected=factory(1).update(strain)[0]
        def task(_):
            native=factory(1)
            for _ in range(5):
                actual,_=native.update(strain)
            return actual
        with ThreadPoolExecutor(max_workers=4) as pool:
            for actual in pool.map(task,range(8)):
                np.testing.assert_allclose(actual,expected,atol=1.e-10)

    def test_cold_initialization_and_process_shutdown(self):
        # Each subprocess has a cold Jem runtime. A first factory constructed
        # by a temporary worker used to produce valid stresses, then crash at
        # process shutdown because its main-thread allocator was missing.
        script = '''
from concurrent.futures import ThreadPoolExecutor
import _fe2_material
def task():
    import numpy as np
    factory=_fe2_material.MaterialFactory(%r)
    return factory(1).update(np.zeros((1,3)))[0]
with ThreadPoolExecutor(max_workers=1) as pool:
    assert (pool.submit(task).result() == 0).all()
''' % ELASTIC
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

        script = '''
from concurrent.futures import ThreadPoolExecutor
import importlib
with ThreadPoolExecutor(max_workers=1) as pool:
    try:
        pool.submit(importlib.import_module,"_fe2_material").result()
    except ImportError as error:
        assert "main thread" in str(error)
    else:
        raise AssertionError("A worker cannot bootstrap the Jem process runtime")
'''
        result=subprocess.run([sys.executable,"-c",script],capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_online_autograd_tangent(self):
        model = network()
        a = np.array([.045,-.004,.01])
        _, tangent = evaluate_with_tangent(model,a)
        h=1.e-7
        fd=np.column_stack([(evaluate_with_tangent(model,a+np.eye(3)[i]*h)[0]-
                             evaluate_with_tangent(model,a-np.eye(3)[i]*h)[0])/(2*h) for i in range(3)])
        np.testing.assert_allclose(tangent,fd,rtol=2.e-5,atol=1.e-5)

    def test_history_gradcheck(self):
        # Earlier plastic loading affects the final unloading stress. A local
        # tangent-only backward cannot pass this full-sequence gradcheck.
        model = network()
        path=torch.tensor([[[.045,-.004,.01],[.006,-.001,.002]]],dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(model,(path,),eps=1.e-6,atol=5.e-4,rtol=1.e-3))
        final_stress=model(path)[:,-1].sum()
        history_gradient=torch.autograd.grad(final_stress,path)[0][0,0]
        self.assertGreater(history_gradient.abs().max().item(),1.)

    def test_encoder_parameter_history_gradient(self):
        model=network()
        path=torch.tensor([[[.04,-.003,.008],[.003,-.001,.001]]],dtype=torch.float64)
        loss=model(path)[:,-1].square().sum()
        loss.backward()
        analytic=model.fc1.weight.grad[0,0].item()
        base=model.fc1.weight[0,0].item()
        h=1.e-5
        with torch.no_grad():
            model.fc1.weight[0,0]=base+h
            plus=model(path)[:,-1].square().sum().item()
            model.fc1.weight[0,0]=base-h
            minus=model(path)[:,-1].square().sum().item()
            model.fc1.weight[0,0]=base
        self.assertAlmostEqual(analytic,(plus-minus)/(2*h),delta=max(.01,abs(analytic)*2.e-4))

    def test_training_isolation_and_scripted_neural_layers(self):
        model=network()
        path=torch.tensor([[[.025,0,.005],[.006,0,.001]]],dtype=torch.float64)
        first=model(path)
        second=model(path)
        torch.testing.assert_close(first,second)
        # Training forward uses an independent context and never resets online history.
        evaluate_with_tangent(model,path[0,0].numpy());model.commit()
        expected=evaluate_with_tangent(model,path[0,1].numpy())[0]
        model(path)
        np.testing.assert_allclose(evaluate_with_tangent(model,path[0,1].numpy())[0],expected,atol=1.e-10)
        other=network().script_neural_layers()
        torch.testing.assert_close(other(path),first)

    def test_nonlinear_encoder_needs_no_cpp_change(self):
        # A test-only architecture substitution proves the C++ driver is opaque
        # to topology; this is not an added production architecture.
        model=network(ELASTIC)
        model.fc1=torch.nn.Sequential(torch.nn.Linear(3,6,bias=False),torch.nn.Tanh()).double()
        model.script_neural_layers()
        a=np.array([.002,-.001,.0003])
        _, tangent=evaluate_with_tangent(model,a)
        h=1.e-7
        fd=np.column_stack([(evaluate_with_tangent(model,a+np.eye(3)[i]*h)[0]-
                             evaluate_with_tangent(model,a-np.eye(3)[i]*h)[0])/(2*h) for i in range(3)])
        np.testing.assert_allclose(tangent,fd,rtol=1.e-5,atol=1.e-5)


if __name__ == '__main__':
    unittest.main()
