class_name Softmax

extends Activation

func apply_activation(inputs: Array, index: int) -> float:
	var max_input = inputs[0]
	for value in inputs:
		if value > max_input:
			max_input = value
	
	var exponential_sum = 0.0
	for value in inputs:
		exponential_sum += exp(value - max_input)
	return exp(inputs[index] - max_input) / exponential_sum

func derivative(inputs: Array, index: int) -> float:
	var value = apply_activation(inputs, index)
	return value * (1 - value)

func get_type():
	return "SOFTMAX"
