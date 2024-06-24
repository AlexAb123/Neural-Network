class_name CrossEntropy

extends Cost

func cost_function(predicted_outputs: Array, expected_outputs: Array) -> float:
	# Expected outputs are all either 0 or 1 -- only one of these in the for loop will be a non-zero number
	var cost = 0.0
	for i in predicted_outputs.size():
		cost += -1 * expected_outputs[i] * log(predicted_outputs[i])
	return cost
	
func derivative(predicted_output: float, expected_output: float) -> float:
	var cost = 0.0
	# This is the formula because when you multiply cross entropy derivative and softmax derivative you want the result to be:
	# predicted_output - expected_output
	# Since softmax doesn't have access to expected_output, the math trickery goes here
	cost += (-predicted_output + expected_output) / (predicted_output * (predicted_output - 1))
	return cost

func get_type():
	return "CROSS_ENTROPY"
