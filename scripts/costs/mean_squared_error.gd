class_name MeanSquaredError

extends Cost

func cost_function(predicted_outputs: Array, expected_outputs: Array) -> float:
	var cost = 0.0
	for i in predicted_outputs.size():
		cost += pow(predicted_outputs[i] - expected_outputs[i], 2)
	return 0.5 * cost
func derivative(predicted_output: float, expected_output: float) -> float:
	return predicted_output - expected_output

func get_type():
	return "MEAN_SQUARED_ERROR"
