shader_type canvas_item;

float random(vec2 uv){
    return fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
}

uniform float angle = 0.0;
uniform vec2 offset = vec2(0.0, 0.0);
uniform float noise_strength = 0.0;

void fragment() {
	// Called for every pixel the material is visible on.
	vec2 uv = UV;

	uv -= vec2(0.5);
	float cos_angle = cos(angle);
	float sin_angle = sin(angle);
	mat2 rotation_matrix = mat2(
		vec2(cos_angle, -sin_angle),
		vec2(sin_angle, cos_angle)
	);
	uv = rotation_matrix * uv;
	uv += vec2(0.5);
	uv += offset;
	vec4 color = texture(TEXTURE, uv);
	float noise = step(0.5, random(uv)) * noise_strength;
	color = color + vec4(noise, noise, noise, 1.0);
	COLOR = color;
}

//void light() {
	// Called for every pixel for every light affecting the CanvasItem.
	// Uncomment to replace the default light processing function with this one.
//}
