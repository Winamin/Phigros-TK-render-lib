use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

const DEFAULT_BUFFER_CAPACITY: usize = 1_000_000;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct NetworkParams {
    pub input_size: u32,
    pub output_size: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub layer_index: u32,
    pub extra_params: u32,
    // Padding fields to match shader's expected 32-byte size
    pub padding0: u32,
    pub padding1: u32,
}

#[derive(Debug)]
struct ExecutorMutableState {
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,
    weights_buffer: wgpu::Buffer,
    biases_buffer: wgpu::Buffer,
    hidden_states_buffer: wgpu::Buffer,
    cell_states_buffer: wgpu::Buffer,
    initial_hidden_buffer: wgpu::Buffer,
    initial_cell_buffer: wgpu::Buffer,
    cached_bind_groups: Option<Vec<wgpu::BindGroup>>,
    weights_uploaded: bool,
    params_data_cache: Vec<NetworkParams>,
    buffer_capacity: usize,
}

#[derive(Debug)]
pub struct GpuNetworkExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    params_buffer: wgpu::Buffer,
    dispatch_buffer: wgpu::Buffer,

    dense_pipeline: wgpu::ComputePipeline,
    lstm_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    residual_pipeline: wgpu::ComputePipeline,
    concat_pipeline: wgpu::ComputePipeline,

    common_bind_group_layout: wgpu::BindGroupLayout,
    lstm_bind_group_layout: wgpu::BindGroupLayout,
    attention_bind_group_layout: wgpu::BindGroupLayout,
    residual_bind_group_layout: wgpu::BindGroupLayout,
    concat_bind_group_layout: wgpu::BindGroupLayout,

    state: Mutex<ExecutorMutableState>,
}

impl GpuNetworkExecutor {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Network Params Buffer"),
            size: std::mem::size_of::<NetworkParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let capacity = DEFAULT_BUFFER_CAPACITY;
        let buffer_size = (capacity * std::mem::size_of::<f32>()) as u64;

        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ping-Pong Buffer A"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ping-Pong Buffer B"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let weights_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Weights Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let biases_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Biases Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hidden_states_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Hidden States Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let cell_states_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Cell States Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dispatch Buffer"),
            contents: bytemuck::cast_slice(&[0u32; 2]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let initial_hidden_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Initial Hidden Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let initial_cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Initial Cell Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let common_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Common Bind Group Layout"),
        });
        let lstm_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: 合并的 uniform 缓冲区 (params + dispatch)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: 合并的 storage 缓冲区 (initial_hidden + initial_cell)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("LSTM Bind Group Layout"),
        });

        let attention_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Attention Bind Group Layout"),
        });

        let residual_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Residual Bind Group Layout"),
        });

        let concat_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("Concat Bind Group Layout"),
        });

        // 创建管线
        let dense_pipeline = create_dense_pipeline(&device, &common_bind_group_layout);
        let lstm_pipeline = create_lstm_pipeline(&device, &lstm_bind_group_layout);
        let attention_pipeline = create_attention_pipeline(&device, &attention_bind_group_layout);
        let residual_pipeline = create_residual_pipeline(&device, &residual_bind_group_layout);
        let concat_pipeline = create_concat_pipeline(&device, &concat_bind_group_layout);

        Self {
            device,
            queue,
            params_buffer,
            dispatch_buffer,
            dense_pipeline,
            lstm_pipeline,
            attention_pipeline,
            residual_pipeline,
            concat_pipeline,
            common_bind_group_layout,
            lstm_bind_group_layout,
            attention_bind_group_layout,
            residual_bind_group_layout,
            concat_bind_group_layout,
            state: Mutex::new(ExecutorMutableState {
                buffer_a,
                buffer_b,
                weights_buffer,
                biases_buffer,
                hidden_states_buffer,
                cell_states_buffer,
                initial_hidden_buffer,
                initial_cell_buffer,
                cached_bind_groups: None,
                weights_uploaded: false,
                params_data_cache: Vec::new(),
                buffer_capacity: capacity,
            }),
        }
    }

    #[inline]
    fn data_buffer_byte_size(element_count: usize) -> u64 {
        (element_count * std::mem::size_of::<f32>()) as u64
    }

    fn make_data_buffer(
        device: &wgpu::Device,
        label: &str,
        byte_size: u64,
        copy_src: bool,
    ) -> wgpu::Buffer {
        let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        if copy_src {
            usage |= wgpu::BufferUsages::COPY_SRC;
        }
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage,
            mapped_at_creation: false,
        })
    }


    pub fn ensure_capacity(&self, required_elements: usize) {
        let mut state = self.state.lock().unwrap();
        if required_elements <= state.buffer_capacity {
            return;
        }
        let new_capacity = required_elements.max(state.buffer_capacity * 2);
        Self::grow_buffers(&self.device, &mut state, new_capacity);
    }

    fn grow_buffers(
        device: &wgpu::Device,
        state: &mut ExecutorMutableState,
        new_capacity: usize,
    ) {
        let new_size = Self::data_buffer_byte_size(new_capacity);
        let old = state.buffer_capacity;

        state.buffer_a = Self::make_data_buffer(device, "Ping-Pong Buffer A", new_size, true);
        state.buffer_b = Self::make_data_buffer(device, "Ping-Pong Buffer B", new_size, true);
        state.weights_buffer = Self::make_data_buffer(device, "Weights Buffer", new_size, false);
        state.biases_buffer = Self::make_data_buffer(device, "Biases Buffer", new_size, false);
        state.hidden_states_buffer = Self::make_data_buffer(device, "Hidden States Buffer", new_size, true);
        state.cell_states_buffer = Self::make_data_buffer(device, "Cell States Buffer", new_size, true);
        state.initial_hidden_buffer = Self::make_data_buffer(device, "Initial Hidden Buffer", new_size, false);
        state.initial_cell_buffer = Self::make_data_buffer(device, "Initial Cell Buffer", new_size, false);

        state.cached_bind_groups = None;
        state.weights_uploaded = false;
        state.buffer_capacity = new_capacity;
        println!("[GPU] Grew data buffers: {} -> {} f32 elements ({} bytes each)",
                 old, new_capacity, new_size);
    }

    pub fn initialize_network(&self, layers: &[NetworkLayerGPU]) {
        let mut state = self.state.lock().unwrap();

        let required = Self::required_capacity_for_layers(layers, 0);
        if required > state.buffer_capacity {
            let new_capacity = required.max(state.buffer_capacity * 2);
            Self::grow_buffers(&self.device, &mut state, new_capacity);
        }

        if !state.weights_uploaded {
            Self::upload_all_weights_and_biases(&self.queue, &state, layers);
            state.weights_uploaded = true;
        }

        state.params_data_cache.clear();
        let mut current_input_size = 0;
        for (i, layer) in layers.iter().enumerate() {
            let params = NetworkParams {
                input_size: current_input_size as u32,
                output_size: layer.output_size as u32,
                batch_size: 1,
                seq_len: layer.seq_len as u32,
                layer_index: i as u32,
                extra_params: match layer.layer_type {
                    LayerTypeGPU::LSTM => (1024u32 << 16) | 0u32,
                    LayerTypeGPU::Dense | LayerTypeGPU::Attention | LayerTypeGPU::Residual => {
                        (1u32 << 8) | 1u32
                    }
                    LayerTypeGPU::Concat => layer.num_inputs as u32,
                },
                padding0: 0,
                padding1: 0,
            };
            state.params_data_cache.push(params);
            current_input_size = layer.output_size;
        }

        let bind_groups = Self::create_ping_pong_bind_groups(
            &self.device, &self.params_buffer,
            &self.common_bind_group_layout, &self.lstm_bind_group_layout,
            &self.attention_bind_group_layout, &self.residual_bind_group_layout,
            &self.concat_bind_group_layout, &state, layers,
        );
        state.cached_bind_groups = Some(bind_groups);
    }

    fn required_capacity_for_layers(layers: &[NetworkLayerGPU], input_len: usize) -> usize {
        let mut max_elem: usize = input_len;
        let mut total_weights = 0usize;
        let mut total_biases = 0usize;
        let mut prev_output = 0usize;
        for layer in layers {
            let input_for_layer = if prev_output == 0 { input_len } else { prev_output };
            max_elem = max_elem.max(input_for_layer).max(layer.output_size);
            total_weights = total_weights.saturating_add(layer.weights_flattened.len());
            total_biases = total_biases.saturating_add(layer.biases.len());
            prev_output = layer.output_size;
        }
        max_elem.max(total_weights).max(total_biases).max(1)
    }

    pub fn execute_network_forward(&self, input_data: &[f32], layers: &[NetworkLayerGPU], output_size: usize) -> Vec<f32> {
        let mut state = self.state.lock().unwrap();

        let required = Self::required_capacity_for_layers(layers, input_data.len())
            .max(output_size);
        if required > state.buffer_capacity {
            let new_capacity = required.max(state.buffer_capacity * 2);
            Self::grow_buffers(&self.device, &mut state, new_capacity);
        }

        if !state.weights_uploaded {
            Self::upload_all_weights_and_biases(&self.queue, &state, layers);
            state.weights_uploaded = true;
        }

        self.queue.write_buffer(&state.buffer_a, 0, bytemuck::cast_slice(input_data));

        if state.cached_bind_groups.is_none() {
            let bind_groups = Self::create_ping_pong_bind_groups(
                &self.device, &self.params_buffer,
                &self.common_bind_group_layout, &self.lstm_bind_group_layout,
                &self.attention_bind_group_layout, &self.residual_bind_group_layout,
                &self.concat_bind_group_layout, &state, layers,
            );
            state.cached_bind_groups = Some(bind_groups);
        }

        let bind_groups = state.cached_bind_groups.as_ref().unwrap();
        Self::execute_layers_optimized(
            &self.device, &self.queue, &self.params_buffer,
            &self.dense_pipeline, &self.lstm_pipeline, &self.attention_pipeline,
            &self.residual_pipeline, &self.concat_pipeline,
            &state, layers, bind_groups,
        );

        let last_buffer = if layers.len() % 2 == 1 {
            &state.buffer_b
        } else {
            &state.buffer_a
        };
        self.download_result_optimized(last_buffer, output_size)
    }

    fn create_ping_pong_bind_groups(
        device: &wgpu::Device,
        params_buffer: &wgpu::Buffer,
        common_layout: &wgpu::BindGroupLayout,
        lstm_layout: &wgpu::BindGroupLayout,
        attention_layout: &wgpu::BindGroupLayout,
        residual_layout: &wgpu::BindGroupLayout,
        concat_layout: &wgpu::BindGroupLayout,
        state: &ExecutorMutableState,
        layers: &[NetworkLayerGPU],
    ) -> Vec<wgpu::BindGroup> {
        let mut bind_groups = Vec::with_capacity(layers.len());

        for (i, layer) in layers.iter().enumerate() {
            let (input_buf, output_buf) = if i % 2 == 0 {
                (&state.buffer_a, &state.buffer_b)
            } else {
                (&state.buffer_b, &state.buffer_a)
            };

            let layout = match layer.layer_type {
                LayerTypeGPU::Dense => common_layout,
                LayerTypeGPU::LSTM => lstm_layout,
                LayerTypeGPU::Attention => attention_layout,
                LayerTypeGPU::Residual => residual_layout,
                LayerTypeGPU::Concat => concat_layout,
            };

            let bind_group = match layer.layer_type {
                LayerTypeGPU::Dense => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: state.biases_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Dense Bind Group", i)),
                    })
                },
                LayerTypeGPU::LSTM => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: state.biases_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: state.hidden_states_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: state.cell_states_buffer.as_entire_binding(),
                            },
                            // binding 6: 合并的 uniform 缓冲区 (params + dispatch)
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: params_buffer.as_entire_binding(),
                            },
                            // binding 7: 合并的 storage 缓冲区 (initial_hidden + initial_cell)
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: state.initial_hidden_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} LSTM Bind Group", i)),
                    })
                },
                LayerTypeGPU::Attention => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: output_buf.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Attention Bind Group", i)),
                    })
                },
                LayerTypeGPU::Residual => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: state.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: state.biases_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Residual Bind Group", i)),
                    })
                },
                LayerTypeGPU::Concat => {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: output_buf.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Concat Bind Group", i)),
                    })
                },
            };

            bind_groups.push(bind_group);
        }

        bind_groups
    }

    fn execute_layers_optimized(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params_buffer: &wgpu::Buffer,
        dense_pipeline: &wgpu::ComputePipeline,
        lstm_pipeline: &wgpu::ComputePipeline,
        attention_pipeline: &wgpu::ComputePipeline,
        residual_pipeline: &wgpu::ComputePipeline,
        concat_pipeline: &wgpu::ComputePipeline,
        state: &ExecutorMutableState,
        layers: &[NetworkLayerGPU],
        bind_groups: &[wgpu::BindGroup],
    ) {
        let mut all_params: Vec<NetworkParams> = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let params = if i < state.params_data_cache.len() {
                state.params_data_cache[i]
            } else {
                NetworkParams {
                    input_size: if i == 0 { 0 } else { layers[i-1].output_size as u32 },
                    output_size: layer.output_size as u32,
                    batch_size: 1,
                    seq_len: layer.seq_len as u32,
                    layer_index: i as u32,
                    extra_params: match layer.layer_type {
                        LayerTypeGPU::LSTM => (1024u32 << 16) | 0u32,
                        LayerTypeGPU::Dense | LayerTypeGPU::Attention | LayerTypeGPU::Residual => {
                            (1u32 << 8) | 1u32
                        }
                        LayerTypeGPU::Concat => layer.num_inputs as u32,
                    },
                    padding0: 0,
                    padding1: 0,
                }
            };
            all_params.push(params);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Optimized Compute Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Optimized Compute Pass"),
                timestamp_writes: None,
            });

            for (i, layer) in layers.iter().enumerate() {
                queue.write_buffer(params_buffer, 0, bytemuck::bytes_of(&all_params[i]));

                // 选择管线
                let pipeline = match layer.layer_type {
                    LayerTypeGPU::Dense => dense_pipeline,
                    LayerTypeGPU::LSTM => lstm_pipeline,
                    LayerTypeGPU::Attention => attention_pipeline,
                    LayerTypeGPU::Residual => residual_pipeline,
                    LayerTypeGPU::Concat => concat_pipeline,
                };

                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_groups[i], &[]);

                // 优化的工作组调度
                let (x, y, z) = Self::calculate_optimal_dispatch(layer.output_size as u32);
                cpass.dispatch_workgroups(x, y, z);
            }
        }

        queue.submit(Some(encoder.finish()));
    }

    #[inline]
    fn calculate_optimal_dispatch(total_items: u32) -> (u32, u32, u32) {
        const WORKGROUP_SIZE: u32 = 256;
        const MAX_DIM: u32 = 65535;

        let total_workgroups = (total_items + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        if total_workgroups <= MAX_DIM {
            (total_workgroups, 1, 1)
        } else if total_workgroups <= MAX_DIM * MAX_DIM {
            let x = (total_workgroups as f32).sqrt().ceil() as u32;
            let y = (total_workgroups + x - 1) / x;
            (x.min(MAX_DIM), y.min(MAX_DIM), 1)
        } else {
            let cube_root = (total_workgroups as f32).powf(1.0 / 3.0).ceil() as u32;
            let x = cube_root.min(MAX_DIM);
            let y = cube_root.min(MAX_DIM);
            let z = ((total_workgroups + x * y - 1) / (x * y)).min(MAX_DIM);
            (x, y, z)
        }
    }

    fn upload_all_weights_and_biases(
        queue: &wgpu::Queue,
        state: &ExecutorMutableState,
        layers: &[NetworkLayerGPU],
    ) {
        let mut all_weights = Vec::new();
        let mut all_biases = Vec::new();

        for layer in layers {
            all_weights.extend_from_slice(&layer.weights_flattened);
            all_biases.extend_from_slice(&layer.biases);
        }

        if !all_weights.is_empty() {
            queue.write_buffer(&state.weights_buffer, 0, bytemuck::cast_slice(&all_weights));
        }

        if !all_biases.is_empty() {
            queue.write_buffer(&state.biases_buffer, 0, bytemuck::cast_slice(&all_biases));
        }
    }

    fn download_result_optimized(&self, source_buffer: &wgpu::Buffer, output_size: usize) -> Vec<f32> {
        let result_size = (output_size * std::mem::size_of::<f32>()) as u64;

        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Download Buffer"),
            size: result_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(source_buffer, 0, &result_buffer, 0, result_size);

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = result_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        let _ = self.device.poll(wgpu::PollType::Wait);

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        result_buffer.unmap();

        result
    }
}

#[derive(Debug, Clone)]
pub struct NetworkLayerGPU {
    pub weights_flattened: Vec<f32>,
    pub biases: Vec<f32>,
    pub output_size: usize,
    pub seq_len: usize,
    pub layer_type: LayerTypeGPU,
    pub num_inputs: usize,
}

#[derive(Debug, Clone)]
pub enum LayerTypeGPU {
    Dense,
    LSTM,
    Attention,
    Residual,
    Concat,
}

fn create_dense_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Dense Pipeline Layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("dense"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/network_compute.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Dense Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn create_lstm_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("LSTM Pipeline Layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lstm"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/lstm.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("LSTM Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn create_attention_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Attention Pipeline Layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("attention"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/attention.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Attention Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn create_residual_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Residual Pipeline Layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("residual"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/residual.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Residual Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}

fn create_concat_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Concat Pipeline Layout"),
        bind_group_layouts: &[layout],
        push_constant_ranges: &[],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("concat"),
        source: wgpu::ShaderSource::Wgsl(include_str!("gpu_shader/concat.wgsl").into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Concat Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    })
}