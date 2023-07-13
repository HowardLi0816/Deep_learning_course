import subprocess
import pytest

EXEC='python3'
SCRIPT_TO_TEST='./hw1p3.py'

# iteration tol
tol = 1e-10
# solution tol
errtol = 1e-6


def run_test_script(args = [], env = {}):
  # check args list
  if not isinstance(args, list):
    args = [args]
  run = [EXEC, SCRIPT_TO_TEST] + args
  print(run)

  # start script
  ps = subprocess.Popen(run, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
  ps.wait()

  print(ps.communicate())

  [stdout, stderr] = [x.decode('utf-8').strip() for x in ps.communicate()]
  returncode = int(ps.returncode)

  return (returncode, stdout, stderr)


def assert_test_output(x0, stdout, stderr):
  if x0 is None:
    # RangeError
    assert (stderr.lower() == 'Range error'.lower()), f'range error, invalid stderr -- stderr:{stderr}'
    assert (stdout == ''), f'range error, invalid stdout -- stdout:{stdout}'
  else:
    # valid
    stdout = stdout.split('\n')

    assert len(stdout) == 4, f'invalid stdout, incorrect line count -- expected:4, actual:{len(stdout)}'

    N = stdout[0]
    assert N.isdigit(), f'expected integer iteration count N -- actual:{N}'

    xn2, xn1, xn0 = stdout[1:4]
    try:
      xn2 = float(xn2)
      xn1 = float(xn1)
      xn0 = float(xn0)
    except ValueError:
      pytest.fail(f'expected float estimates -- xn2:{xn2}, xn1:{xn1}, xn0:{xn0}')

    assert abs(xn2 - xn1) >= tol, f'extra iteration detected -- xn2:{xn2}, xn1:{xn1}, tol:{tol}, delta:{abs(xn2-xn1)}'
    assert abs(xn1 - xn0) < tol, f'step tolerance exceeded -- xn1:{xn1}, xn0:{xn0}, tol:{tol}, delta:{abs(xn1-xn0)}'

    assert abs(xn0 - x0) < errtol, f'estimate error exceeds tolerance -- xn0:{xn0}, x0:{x0}, tol:{errtol}'

@pytest.mark.parametrize('a, b, x0', [(1.0, 2.0, 1.46557123187676802665673122522), (2.0, 1.0, None)])
def test_f1(a, b, x0):
  env = {'FN':'f1'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)


@pytest.mark.parametrize('a, b, x0', [(-0.7, -0.6, -0.659266045766946074537348579563), (-1.3, -1.2, -1.27102680081594606400471884810)])
def test_f2(a, b, x0):
  env = {'FN':'f2'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)


@pytest.mark.parametrize('a, b, x0', [(1.0, 2.0, 1.57079632679489661923132169164)])
def test_f3(a, b, x0):
  env = {'FN':'f3'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)


@pytest.mark.parametrize('a, b, x0', [(0.445, 0.449, 0.447431543288746570049221830350)])
def test_f4(a, b, x0):
  env = {'FN':'f4'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)


@pytest.mark.parametrize('a, b, x0', [(2.5, 2.8, 2.66666666666666691338289436115)])
def test_f5(a, b, x0):
  env = {'FN':'f5'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)


@pytest.mark.parametrize('a, b, x0', [(-0.1, 0.0, -0.0202026931403341150199824288113)])
def test_f6(a, b, x0):
  env = {'FN':'f6'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(x0, stdout, stderr)

@pytest.mark.parametrize('a, b', [(2.0, 1.0)])
def test_invalid1(a, b):
  env = {'FN':'f1'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(None, stdout, stderr)

@pytest.mark.parametrize('a, b', [(-0.5, 0.0)])
def test_invalid2(a, b):
  env = {'FN':'f1'}
  args = [str(a), str(b)]
  _, stdout, stderr = run_test_script(args=args, env=env)
  assert_test_output(None, stdout, stderr)
