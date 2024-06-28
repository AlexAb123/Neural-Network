shader_type canvas_item;

void vertex() {
	// Called for every vertex the material is visible on.
}

uniform float angle = 360.0;

void fragment() {
	// Called for every pixel the material is visible on.
	//COLOR = vec4(UV, 0.5, 1.0);
	//float sine = sin(angle);
	//float cosine = cos(angle);
	
	
    vec2 uv = UV;
    uv -= vec2(0.5); // Center the UV coordinates
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    
	mat2 rotation_matrix = mat2(
		vec2(cos_angle, -sin_angle),
		vec2(sin_angle, cos_angle)
	);
    
    uv = rotation_matrix * uv;
    uv += vec2(0.5); // Translate back
    vec4 color = texture(TEXTURE, uv);
    COLOR = color;
}

//void light() {
	// Called for every pixel for every light affecting the CanvasItem.
	// Uncomment to replace the default light processing function with this one.
//}
