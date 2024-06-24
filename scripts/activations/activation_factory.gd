class_name ActivationFactory

extends Resource

enum type {
	SIGMOID,
	RELU,
	LEAKY_RELU,
	SOFTMAX
}

static func new_activation(activation_type: type = type.RELU):
	match activation_type:
		type.SIGMOID:
			return Sigmoid.new()
		type.RELU:
			return Relu.new()
		type.LEAKY_RELU:
			return LeakyRelu.new()
		type.SOFTMAX:
			return Softmax.new()

static func get_type_by_name(name: String):
	match name:
		"SIGMOID":
			return ActivationFactory.type.SIGMOID
		"RELU":
			return ActivationFactory.type.RELU
		"LEAKY_RELU":
			return ActivationFactory.type.LEAKY_RELU
		"SOFTMAX":
			return ActivationFactory.type.SOFTMAX
