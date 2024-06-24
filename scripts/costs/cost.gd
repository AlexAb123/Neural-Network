class_name Cost

extends Resource

func cost_function(predicted_outputs: Array, expected_outputs: Array) -> float:
	return 0.0
	
func derivative(predicted_output: float, expected_output: float) -> float:
	return 0.0

func get_type():
	return "COST"
