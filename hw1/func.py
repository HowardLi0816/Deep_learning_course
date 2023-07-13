import math
import os

def f(x):
  fn = os.getenv('FN', 'f1')  # default f1
  return getattr(FNS, fn)(x)

 
class FNS:
  def f1(x):
    return x**3 - x**2 - 1
  def f2(x):
    return math.cos(x) + 2 * math.sin(x) + x**2
  def f3(x):
    return math.exp(-x) * math.cos(x)
  def f4(x):
    return math.tan(x * math.pi) - 6
  def f5(x):
    return (0.8 - 0.3 * x) / x
  def f6(x):
    return math.sin(x**2) + 1.02 - math.exp(-x)
