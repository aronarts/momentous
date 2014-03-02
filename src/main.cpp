#define GLEW_STATIC
#include <GL/glew.h>

#include <GLFW/glfw3.h>

#include <array>
#include <thread>
#include <stdio.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define STRINGIZE2(a) STRINGIZE(a)
#define STRINGIZE(a) #a

#define CS_LOCAL_SIZE_X 256

namespace 
{
	extern const char * update_pos_cs_source[];
	extern const char * update_vel_cs_source[];
	extern const char * cube_vs[];
	extern const char * cube_fs[];

	const unsigned int kNumCubes = 48 * 1024;
	const float part_size = 0.001f;
	const GLfloat zero = 0;
}

static void APIENTRY debug_callback(GLenum source,
	GLenum type,
	GLuint id,
	GLenum severity,
	GLsizei length,
	const GLchar* message,
	GLvoid* userParam)
{
	fprintf(stderr, "OpenGL message: %s\n", message);

	if (GL_DEBUG_TYPE_ERROR == type) 
		exit(1);
}

static bool is_pow2(int x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

static float randf()
{
	return 1.0f * rand() / RAND_MAX;
}

static glm::vec3 rand_vec3_unit_sphere(float* len_sq_out = nullptr)
{
	glm::vec3 v;
	float l;

	do
	{
		v.x = 2.0f * randf() - 1.0f;
		v.y = 2.0f * randf() - 1.0f;
		v.z = 2.0f * randf() - 1.0f;
		l = glm::length(v);
	} while (l > 1.0f);

	if (len_sq_out)
		*len_sq_out = l;

	return v;
}

static glm::vec3 rand_unit_vec3()
{
	glm::vec3 v;
	float l;

	do v = rand_vec3_unit_sphere(&l); while (l == 0.0f);
	return glm::inversesqrt(l) * v;
}

static int step_idx(int base, int step, int mask)
{
	return (base & ~mask) | ((base + step) & mask);
}

static GLuint make_force_tex(int size, float strength, float post_scale)
{
	using namespace glm;
	assert(is_pow2(size));

	int stepx = 1, maskx = size - 1;
	int stepy = size, masky = (size - 1) * size;
	int stepz = size*size, maskz = (size - 1) * size * size;
	int nelem = size * size * size;
	vec4* forces = new vec4[nelem];

	// create a random vector field
	for (int zo = 0; zo <= maskz; zo += stepz) {
		for (int yo = 0; yo <= masky; yo += stepy) {
			for (int xo = 0; xo <= maskx; xo += stepx) {
				forces[xo + yo + zo] = vec4(strength * rand_unit_vec3(), 0.0f);
			}
		}
	}

	// calc divergences
	float* div = new float[nelem];
	float* high = new float[nelem];

	float div_scale = -0.5f / (float)size;

	for (int zo = 0; zo <= maskz; zo += stepz) {
		for (int yo = 0; yo <= masky; yo += stepy) {
			for (int xo = 0; xo <= maskx; xo += stepx) {
				int o = xo + yo + zo;

				div[o] = div_scale *
					(
					forces[step_idx(o, stepx, maskx)].x - forces[step_idx(o, -stepx, maskx)].x +
					forces[step_idx(o, stepy, masky)].y - forces[step_idx(o, -stepy, masky)].y +
					forces[step_idx(o, stepz, maskz)].z - forces[step_idx(o, -stepz, maskz)].z
					);
				high[o] = 0.0f;
			}
		}
	}

	// gauss-seidel iteration to calc density field
	for (int step = 0; step < 40; step++) {
		for (int zo = 0; zo <= maskz; zo += stepz) {
			for (int yo = 0; yo <= masky; yo += stepy) {
				for (int xo = 0; xo <= maskx; xo += stepx) {
					int o = xo + yo + zo;
					high[o] =
						(
						high[step_idx(o, -stepx, maskx)] + high[step_idx(o, stepx, maskx)] +
						high[step_idx(o, -stepy, masky)] + high[step_idx(o, stepy, masky)] +
						high[step_idx(o, -stepz, maskz)] + high[step_idx(o, stepz, maskz)]
						) * (1.0f / 6.0f) - div[o];
				}
			}
		}
	}

	// remove gradients from vector field
	float grad_scale = 0.5f * (float)size;
	for (int zo = 0; zo <= maskz; zo += stepz) {
		for (int yo = 0; yo <= masky; yo += stepy) {
			for (int xo = 0; xo <= maskx; xo += stepx) {
				int o = xo + yo + zo;
				vec4* f = forces + o;

				f->x = (f->x - grad_scale * (high[step_idx(o, stepx, maskx)] - high[step_idx(o, -stepx, maskx)])) * post_scale;
				f->y = (f->y - grad_scale * (high[step_idx(o, stepy, masky)] - high[step_idx(o, -stepy, masky)])) * post_scale;
				f->z = (f->z - grad_scale * (high[step_idx(o, stepz, maskz)] - high[step_idx(o, -stepz, maskz)])) * post_scale;
			}
		}
	}

	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_3D, tex);
	glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA32F, size, size, size);
	glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, size, size, size, GL_RGBA, GL_FLOAT, forces);

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

	delete[] div;
	delete[] high;
	delete[] forces;

	return tex;
}

struct UpdateConstBuf
{
	float damping; float accel; float vel_scale; float std140_pad0;
	glm::vec4 field_scale;
	glm::vec4 field_offs;
	glm::vec4 field_sample_scale;
};

void glfw_callback(int i, const char* c)
{
	printf("GLFW: %s\n", c);
}

int main(int argc, char *argv[])
{
	GLFWwindow* window;

	if (!glfwInit())
	{
		fprintf(stderr, "glfwInit failed\n");
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);


	glfwSwapInterval(1.0);
	glfwSetErrorCallback(glfw_callback);

	window = glfwCreateWindow(1280, 720, "Hello World", NULL, NULL);
	if (!window)
	{
		fprintf(stderr, "glfwCreateWindow failed\n");
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		fprintf(stderr, "glfwInit failed\n");
		return -1;
	}

	printf("OpenGL %s\n", glGetString(GL_VERSION));
	printf("GLSL %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	srand(0);

	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
	glDebugMessageCallbackARB(debug_callback, NULL);

	GLuint cube_draw_program = glCreateProgram();
	{
		GLuint vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, cube_vs, NULL);
		glCompileShader(vs);

		GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, cube_fs, NULL);
		glCompileShader(fs);

		glAttachShader(cube_draw_program, vs);
		glAttachShader(cube_draw_program, fs);
		glLinkProgram(cube_draw_program);

		glDeleteShader(vs);
		glDeleteShader(fs);
	}

	GLuint update_pos_cs = glCreateProgram();
	{
		GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
		glShaderSource(cs, 1, update_pos_cs_source, NULL);
		glCompileShader(cs);

		glAttachShader(update_pos_cs, cs);
		glLinkProgram(update_pos_cs);

		glDeleteShader(cs);
	}

	GLuint constant_buffer;
	{
		glGenBuffers(1, &constant_buffer);
		glBindBuffer(GL_UNIFORM_BUFFER, constant_buffer);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(UpdateConstBuf), NULL, GL_STATIC_DRAW);

		auto update_consts = (UpdateConstBuf*)glMapBuffer(GL_UNIFORM_BUFFER, GL_WRITE_ONLY);
		update_consts->field_scale = glm::vec4(32.0f);
		update_consts->damping = 0.99f;
		update_consts->field_offs = glm::vec4(0.0f);
		update_consts->accel = 0.75f;
		update_consts->field_sample_scale = glm::vec4(1.0f / 32.0f);
		update_consts->vel_scale = part_size * 6.0f;
		glUnmapBuffer(GL_UNIFORM_BUFFER);

		glBindBufferBase(GL_UNIFORM_BUFFER, 0, constant_buffer);
	}

	// triple-buffer for position, plus velocity
	std::array<GLuint, 4> part_tex;
	{
		glGenBuffers(4, part_tex.data());

		for (int i = 0; i < 4; i++)
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, part_tex[i]);
			glBufferData(GL_SHADER_STORAGE_BUFFER, kNumCubes * sizeof(glm::vec4), NULL, GL_DYNAMIC_COPY);

			// Zero buffer
			glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &zero);

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	}

	GLuint force_tex = make_force_tex(32, 1.0f, 0.001f);

	GLuint shader_data_buffer;
	{
		glGenBuffers(1, &shader_data_buffer);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, shader_data_buffer);
		glBufferData(GL_SHADER_STORAGE_BUFFER, kNumCubes * sizeof(glm::mat4), NULL, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	GLuint cube_vao, cube_ibo;
	{
		// Create VAO
		glGenVertexArrays(1, &cube_vao);
		glBindVertexArray(cube_vao);

		glGenBuffers(1, &cube_ibo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ibo);

		static const GLushort cube_inds[] = {
			0, 2, 1, 3, 7, 2, 6, 0, 4, 1, 5, 7, 4, 6,
		};
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(cube_inds), cube_inds, GL_STATIC_DRAW);

	}

	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_FRAMEBUFFER_SRGB);
	glFrontFace(GL_CCW);

	glClearColor(0.2f, 0.4f, 0.6f, 1.0f);

	GLint cubePVLocation = glGetUniformLocation(cube_draw_program, "mvp");

	glm::vec3 world_cam_pos(0.0f, 0.0f, -0.9f);
	glm::mat4 clip_from_view = glm::perspective(60.0f, 1.77f, 0.01f, 50.0f);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, force_tex);

	glBindVertexArray(cube_vao);

	int frame = 0;
	unsigned int cur_part = 0;
	unsigned int spawn_counter = 0;

	while (!glfwWindowShouldClose(window))
	{
		using namespace glm;
				
		vec3 emit_pos(0.0f);
		emit_pos.x = 0.7f * sin(frame * 0.001f);

		// spawn new particles
		static const int kSpawnCount = 256;
		{
			vec4 pos_old[kSpawnCount];
			vec4 pos_new[kSpawnCount];

			for (int i = 0; i < kSpawnCount; i++)
			{
				vec3 pos = emit_pos + rand_vec3_unit_sphere() * 0.002f;
				vec3 vel = rand_vec3_unit_sphere() * 0.003f;

				pos_old[i] = vec4(pos - vel, part_size);
				pos_new[i] = vec4(pos, part_size);
			}

			assert(spawn_counter + kSpawnCount - 1 < kNumCubes);

#define DO_MAP_BUFFER 0 // Mapping is quite a bit slower
#if DO_MAP_BUFFER
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, part_tex[(cur_part + 2) % 3]);
			auto m = (GLfloat*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, spawn_counter * sizeof(vec4), kSpawnCount * sizeof(vec4), GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT);
			memcpy(m, &pos_old[0], kSpawnCount * sizeof(vec4));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

			glBindBuffer(GL_SHADER_STORAGE_BUFFER, part_tex[cur_part]);
			m = (GLfloat*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, spawn_counter * sizeof(vec4), kSpawnCount * sizeof(vec4), GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT);
			memcpy(m, &pos_new[0], kSpawnCount * sizeof(vec4));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
#else
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, part_tex[(cur_part + 2) % 3]);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, spawn_counter * sizeof(vec4), kSpawnCount * sizeof(vec4), pos_old);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, part_tex[cur_part]);
			glBufferSubData(GL_SHADER_STORAGE_BUFFER, spawn_counter * sizeof(vec4), kSpawnCount * sizeof(vec4), pos_new);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#endif

			spawn_counter = (spawn_counter + kSpawnCount) % kNumCubes;
		}

		// update position (potentially several time steps)
		for (int step = 0; step < 1; step++)
		{
			cur_part = (cur_part + 1) % 3;

			GLuint buffers[2];

			for (int i = 0; i < 2; i++)
				buffers[i] = part_tex[(cur_part + 1 + i) % 3];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[0]); // older pos
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[1]); // newer pos
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, part_tex[cur_part]); // output

			glUseProgram(update_pos_cs);
			glDispatchCompute(kNumCubes / CS_LOCAL_SIZE_X, 1, 1);

			glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		}

		// set up camera
		vec3 world_cam_target = emit_pos;
		mat4 view_from_world = glm::lookAt(world_cam_pos, world_cam_target, vec3(0, -1, 0));
		mat4 clip_from_world = clip_from_view * view_from_world;

		GLuint buffers[2];

		for (int i = 0; i < 2; i++)
			buffers[i] = part_tex[(cur_part + 2 + i) % 3];

		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffers[0], 0, kNumCubes * sizeof(glm::vec4)); // older pos
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 1, buffers[1], 0, kNumCubes * sizeof(glm::vec4)); // newer pos
		glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 6, shader_data_buffer, 0, kNumCubes * sizeof(glm::mat4));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(cube_draw_program);
		glUniformMatrix4fv(cubePVLocation, 1, GL_FALSE, (const GLfloat*) &clip_from_world);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_ibo);
		glDrawElementsInstanced(GL_TRIANGLE_STRIP, 15, GL_UNSIGNED_SHORT, NULL, kNumCubes);

		glfwSwapBuffers(window);
		glfwPollEvents();

		frame++;

		//std::this_thread::sleep_for(std::chrono::milliseconds(32));
	}

	glDeleteProgram(cube_draw_program);
	glDeleteProgram(update_pos_cs);

	glDeleteBuffers(1, &cube_ibo);
	glDeleteBuffers(1, &constant_buffer);
	glDeleteBuffers(1, &shader_data_buffer);
	glDeleteBuffers(4, part_tex.data());

	glDeleteTextures(1, &force_tex);

	glDeleteVertexArrays(1, &cube_vao);

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}

namespace
{
const char * update_pos_cs_source[] =
{
	"#version 430 core\n"

	"layout (local_size_x = " STRINGIZE2(CS_LOCAL_SIZE_X) ") in;\n"

STRINGIZE(
	uniform sampler3D tex_force;

	layout(binding = 0, std140) uniform UpdateConsts
	{
		float damping;
		float accel;
		float vel_scale;
		vec4 field_scale;
		vec4 field_offs;
		vec4 field_sample_scale;
	};

	layout(binding = 1, std430) buffer block1
	{
		vec4 older_position[];
	};

	layout(binding = 2, std430) buffer as
	{
		vec4 newer_position[];
	};

	layout(binding = 3, std430) buffer block3
	{
		vec4 pos_out[];
	};

	void main(void)\n
	{
		uint gid = gl_GlobalInvocationID.x;\n
		vec4 older_pos = older_position[gid];\n
		vec4 newer_pos = newer_position[gid];\n

		// determine force field sample pos
		vec3 force_pos = newer_pos.xyz * field_scale.xyz + field_offs.xyz;\n
		vec3 force_frac = fract(force_pos);\n
		vec3 force_smooth = force_frac * force_frac * (3.0 - 2.0 * force_frac);\n
		force_pos = (force_pos - force_frac) + force_smooth;\n

		// sample force from texture
		vec3 force = texture(tex_force, force_pos * field_sample_scale.xyz).xyz;\n

		// verlet integration
		vec3 new_pos = newer_pos.xyz + damping * (newer_pos.xyz - older_pos.xyz);\n
		new_pos += accel * force;\n

		vec4 o = vec4(new_pos, newer_pos.w);\n

		// nuke particles if they get too far from the origin
		if (dot(new_pos, new_pos) > 16.0)\n
			o.w = 0.0;\n

		pos_out[gid] = o;\n
	}
)
};

const char* cube_vs[] = 
{
	"#version 430 core\n"

STRINGIZE(
	out vec3 world_pos;

	uniform mat4 mvp;

	layout(binding = 0, std430) buffer block0
	{
		vec4 older_position[];
	};\n

	layout(binding = 1, std430) buffer block1
	{
		vec4 newer_position[];
	};\n
		
	void main() 
	{
		uint gid = gl_InstanceID;
		vec4 older_pos = older_position[gid];
		vec4 newer_pos = newer_position[gid];

		if (newer_pos.w == 0.0) 
		{
			gl_Position = vec4(0, 0, 0, 0);
			world_pos = vec3(0, 0, 0);
			return;
		}

		vec3 forward = vec3(newer_pos - older_pos);

		const vec3 world_down = vec3(0, 1, 0);

		vec3 X = forward;
		vec3 Z = normalize(cross(X, world_down));
		vec3 Y = normalize(cross(Z, X));

		float across_size = newer_pos.w;

		world_pos = newer_pos.xyz;
		world_pos += (((gl_VertexID & 1) != 0) ? 1.0 : -1.0) * X;
		world_pos += (((gl_VertexID & 2) != 0) ? across_size : -across_size) * Y;
		world_pos += (((gl_VertexID & 4) != 0) ? across_size : -across_size) * Z;

		gl_Position = mvp * vec4(world_pos, 1.0);
	}
)
};

const char* cube_fs[] = 
{
	"#version 430 core\n"

STRINGIZE(
	in vec3 world_pos;
	out vec4 outColor;

	const vec3 light_direction = normalize(vec3(0.0, 0.7, 0.3));

	void main()
	{
		// determine triangle plane from derivatives
		vec3 dPos_dx = dFdx(world_pos.xyz);
		vec3 dPos_dy = dFdy(world_pos.xyz);

		// world-space normal from tangents
		vec3 normal_world = normalize(cross(dPos_dy, dPos_dx));

		float NdotL = dot(normal_world.xyz, light_direction);
		float clampNdotL = max(0.0, NdotL);

		vec3 diffuse_lit = vec3(0.0144438436)
			+ clampNdotL * vec3(0.527115226)
			+ (1.0 - abs(NdotL)) * vec3(0.116970673, vec2(0.0144438436))
			+ max(-NdotL, 0.0) * vec3(vec2(0.00518151699), 0.0512694679);

		outColor = vec4(diffuse_lit, 1.0);
	}
)
};
}