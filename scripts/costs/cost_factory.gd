class_name CostFactory

extends Resource

enum type {
	CROSS_ENTROPY,
	MEAN_SQUARED_ERROR
}

static func new_cost(cost_type: type = type.CROSS_ENTROPY):
	match cost_type:
		type.CROSS_ENTROPY:
			return CrossEntropy.new()
		type.MEAN_SQUARED_ERROR:
			return MeanSquaredError.new()

static func get_type_by_name(name: String):
	match name:
		"CROSS_ENTROPY":
			return CostFactory.type.CROSS_ENTROPY
		"MEAN_SQUARED_ERROR":
			return CostFactory.type.MEAN_SQUARED_ERROR
