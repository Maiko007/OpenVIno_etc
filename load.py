model_path ='MODEL_PATH'

#IECore version
from openvino.inference_engine import IENetwork,IECore
ie = IECore()
net  = ie.read_network(model_path+'.xml', model_path+'.bin')
input_name  = next(iter(net.inputs))
input_shape = net.inputs[input_name].shape
out_name    = next(iter(net.outputs))
out_shape   = net.outputs[out_name].shape
exec_net    = ie.load_network(net, 'YOUR_DEVICE')
###

#In case of error, use the following code

#IEPlugin version
from openvino.inference_engine import IENetwork,IEPlugin
plugin = IEPlugin(device='YOUR_DEVICE')
net = IENetwork(model= model_path + ".xml",weights= model_path + ".bin")
input_name  = next(iter(net.inputs))
input_shape = net.inputs[input_name].shape
out_name    = next(iter(net.outputs))
out_shape   = net.outputs[out_name].shape
exec_net    = plugin.load(network=net)
###
