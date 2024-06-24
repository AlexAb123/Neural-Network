class_name Layer

extends Resource

var input_count: int
var output_count: int

# 2D array. Has the same amount of rows as output_count. Has the same amount of columns as input_count.
var weights: Array[Array]
var biases: Array

var inputs: Array
var weighted_sums: Array
var activation: Activation
var outputs: Array

var deltas: Array

# 2D array. Has the same amount of rows as output_count. Has the same amount of columns as input_count.
var weight_gradients: Array[Array]
var bias_gradients: Array
var prev_weight_gradients: Array[Array]
var prev_bias_gradients: Array

func rr(min: float, max: float) -> float:
	return min + (max - min) * randi() / pow(2,32)
func _init(_input_count = 0, _output_count = 0, _activation = null):
	input_count = _input_count
	output_count = _output_count
	activation = _activation
	var temp_weights = []
	var temp_weight_gradients = []
	var temp_prev_weight_gradients = []
 
	for i in output_count:
		temp_weights = []
		temp_weight_gradients = []
		temp_prev_weight_gradients = []
		biases.append(0.0)
		bias_gradients.append(0.0)
		prev_bias_gradients.append(0.0)
		deltas.append(0.0)
		weighted_sums.append(0.0)
		outputs.append(0.0)
		for j in input_count:
			var r = rr(-1,1)
			temp_weights.append(r)
			temp_weight_gradients.append(0.0)
			temp_prev_weight_gradients.append(0.0)
		weights.append(temp_weights)
		weight_gradients.append(temp_weight_gradients)
		prev_weight_gradients.append(temp_prev_weight_gradients)
	for i in input_count:
		inputs.append(0.0)
		
func calculate_weighted_sums(data_point: Array):
	inputs = data_point
	for node_out in output_count:
		var sum = biases[node_out]
		for node_in in input_count:
			sum += weights[node_out][node_in] * data_point[node_in]
		weighted_sums[node_out] = sum
	
func calculate_output(data_point: Array):
	calculate_weighted_sums(data_point)
	for i in weighted_sums.size():
		outputs[i] = activation.apply_activation(weighted_sums, i)
	return outputs
	
func calculate_output_layer_deltas(label: Array, cost: Cost):
	if activation.get_type() == "SOFTMAX" and cost.get_type() == "CROSS_ENTROPY":
		for node_out in output_count:
			deltas[node_out] = outputs[node_out] - label[node_out]
	else:
		for node_out in output_count:
			deltas[node_out] = activation.derivative(weighted_sums, node_out) * cost.derivative(outputs[node_out], label[node_out])
		
func calculate_hidden_layer_deltas(next_layer: Layer):
	for i in deltas.size():
		deltas[i] = 0.0
	for current_layer_node in output_count:
		for next_layer_node in next_layer.output_count:
			deltas[current_layer_node] += next_layer.deltas[next_layer_node] * next_layer.weights[next_layer_node][current_layer_node]
	for i in output_count:
		deltas[i] *= activation.derivative(weighted_sums, i)
		
func update_gradients():
	for node_out in output_count:
		for node_in in input_count:
			# Acculumate because we will reset it when we apply it (which will be after every batch, which we take the average with respect to)
			weight_gradients[node_out][node_in] += deltas[node_out] * inputs[node_in]
			
	for node_out in output_count:
		bias_gradients[node_out] += deltas[node_out]
		
func apply_gradients(learn_rate: float, batch_size: int, momentum: float):
	
	for node_out in output_count:
		for node_in in input_count:
			var step = (learn_rate * weight_gradients[node_out][node_in] + momentum * prev_weight_gradients[node_out][node_in]) / batch_size
			weights[node_out][node_in] -= step
			prev_weight_gradients[node_out][node_in] = weight_gradients[node_out][node_in]
			
	for node_out in output_count:
		var step = (learn_rate * bias_gradients[node_out] + momentum * prev_bias_gradients[node_out]) / batch_size
		biases[node_out] -= step
		prev_bias_gradients[node_out] = bias_gradients[node_out]
	
	reset_gradients()
	
func reset_gradients():
	for node_out in output_count:
		bias_gradients[node_out] = 0.0
		for node_in in input_count:
			weight_gradients[node_out][node_in] = 0.0

func save_as_dict():
	return {
		"biases": biases,
		"weights": weights,
		"input_count": input_count,
		"output_count": output_count,
		"weight_gradients": weight_gradients,
		"prev_weight_gradients": prev_weight_gradients,
		"bias_gradients": bias_gradients,
		"prev_bias_gradients": prev_bias_gradients,
		"deltas": deltas,
		"weighted_sums": weighted_sums,
		"outputs": outputs,
		"activation": activation.get_type(),
	}
	
func load_from_dict(data: Dictionary):
	input_count = data["input_count"]
	output_count = data["output_count"]
	weights.assign(data["weights"])
	weight_gradients.assign(data["weight_gradients"])
	prev_weight_gradients.assign(data["prev_weight_gradients"])
	biases = data["biases"]
	bias_gradients = data["bias_gradients"]
	prev_bias_gradients = data["prev_bias_gradients"]
	deltas = data["deltas"]
	weighted_sums = data["weighted_sums"]
	outputs = data["outputs"]
	activation = ActivationFactory.new_activation(ActivationFactory.get_type_by_name(data["activation"]))

func _to_string():
	var string = "Weights:\n"
	for node_out in output_count:
		for node_in in input_count:
			string += "[" + str(node_in) + " -> " + str(node_out)+ "] " + str(weights[node_out][node_in]) + "\n"
		string += "\n"
	#string += "Weight Gradients:\n"
	#for node_out in output_count:
		#for node_in in input_count:
			#string += "[" + str(node_in) + " -> " + str(node_out)+ "] " + str(snapped(weight_gradients[node_out][node_in], 0.01)) + "\n"
		#string += "\n"
	string += "Biases:\n"
	for node_out in output_count:
		string += "[" + str(node_out) + "] " + str(biases[node_out]) + "\n"
	#string += "\nBias Gradients:\n"
	#for node_out in output_count:
		#string += "[" + str(node_out) + "] " + str(bias_gradients[node_out]) + "\n"
	string += "\nDeltas:\n"
	for node_out in output_count:
		string += "[" + str(node_out) + "] " + str(deltas[node_out]) + "\n"
	return string
