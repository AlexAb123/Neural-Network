class_name LeakyRelu

extends Activation

func apply_activation(inputs: Array, index: int) -> float:
	return inputs[index] if inputs[index] > 0 else 0

func derivative(inputs: Array, index: int) -> float:
	return 1 if inputs[index] < 0 else 0.0001 * inputs[index]

func get_type():
	return "LEAKY_RELU"
