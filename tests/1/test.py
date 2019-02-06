import tth_object
import sys

samples = None
n_epochs = None
node_arch = None

for i, arg in enumerate(sys.argv):
    if arg == '-s':
        samples = int(sys.argv[i + 1])
    if arg == '-ne':
        n_epochs = int(sys.argv[i + 1])
    if arg == '-na':
        node_arch = tuple(int(elt) for elt in sys.argv[i + 1].split(","))
    if arg == '-n':
        name = sys.argv[i + 1]

if samples is None:
    samples = 25000
if n_epochs is None: 
    n_epochs = 10
if node_arch is None:
    node_arch = tuple(10,10)
if name is None: 
    name = "carl"

# node_arch = tuple(int(sys.argv[i + 1]) for i in range(3))
# n_epochs = int(sys.argv[4])
# samples = int(sys.argv[5])
# print(samples, type(samples))
# print(n_epochs, type(n_epochs))
# print(node_arch, type(node_arch))

t = tth_object.new_tth_object(samples=samples)

tth_object.train_and_evaluate(t,
                              name=name,
                              node_arch=node_arch,
                              n_epochs=n_epochs)
