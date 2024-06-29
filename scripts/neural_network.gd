class_name NeuralNetwork

extends Resource

var layers: Array[Layer] = []

var cost: Cost

func _init(layer_sizes: Array = [], hidden_layer_activation = null, output_layer_activation = null, _cost = null):
	cost = _cost
	for i in range(1, layer_sizes.size()):
		var a: Activation = hidden_layer_activation if i < layer_sizes.size() - 1 else output_layer_activation
		layers.append(Layer.new(layer_sizes[i-1], layer_sizes[i], a))

func forward_propagate(data_point: Array):
	var output: Array = data_point
	for layer in layers:
		output = layer.calculate_output(output)
	return output
	
func back_propagate(label: Array):
	layers[layers.size() - 1].calculate_output_layer_deltas(label, cost)
	for i in range(layers.size() - 2, -1, -1):
		layers[i].calculate_hidden_layer_deltas(layers[i+1])
	update_all_gradients()
	
func update_all_gradients():
	# Accumulate gradients for each layer
	for layer in layers:
		layer.update_gradients()

func apply_all_gradients(learn_rate: float, batch_size: int, momentum: float):
	for layer in layers:
		layer.apply_gradients(learn_rate, batch_size, momentum)

func train(data: Array, labels: Array, learn_rate: float, momentum: float):
	for i in data.size():
		var data_point = data[i]
		var label = labels[i]
		forward_propagate(data_point)
		back_propagate(label)
	apply_all_gradients(learn_rate, data.size(), momentum)

func calculate_average_cost(data: Array, labels: Array):
	var total_cost = 0.0
	for i in data.size():
		var prediction = forward_propagate(data[i])
		total_cost += calculate_cost(prediction, labels[i])
	return total_cost / data.size()
	
func calculate_cost(prediction, label):
	return cost.cost_function(prediction, label)
	
func save_to_file(file_path: String):
	var data = {
		"layers": [],
		"cost": cost.get_type()
	}
	for layer in layers:
		data["layers"].append(layer.save_as_dict())

	var file = FileAccess.open(file_path, FileAccess.WRITE)
	var json = JSON.new()
	file.store_string(json.stringify(data))
	file.close()

static func load_from_file(file_path: String):
	
	var net = NeuralNetwork.new()
	var file = FileAccess.open(file_path, FileAccess.READ)
	var json = JSON.new()
	var data = json.parse_string(file.get_as_text())
	file.close()
	
	net.cost = CostFactory.new_cost(CostFactory.get_type_by_name(data["cost"]))

	net.layers.clear()
	for layer_data in data["layers"]:
		var layer = Layer.new()
		layer.load_from_dict(layer_data)
		net.layers.append(layer)
	
	return net

func _to_string():
	var string = ""
	for i in layers.size():
		string += "Layer " + str(i) + ":\n" + str(layers[i]) + "\n----------------------------------------------\n"
	return string


