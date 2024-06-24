class_name Sigmoid

extends Activation

func apply_activation(inputs: Array, index: int) -> float:
	return 1.0 / (1.0 + exp(-inputs[index]))

func derivative(inputs: Array, index: int) -> float:
	var s = 1.0 / (1.0 + exp(-inputs[index]))
	return s * (1.0 - s)

func get_type():
	return "SIGMOID"
