use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

// 为GPU实现创建统一的缓冲区管理
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

// 统一的GPU网络执行器
#[derive(Debug)]
pub struct GpuNetworkExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    // 统一的参数缓冲区
    params_buffer: wgpu::Buffer,

    // Ping-Pong缓冲区策略：在层之间交替使用，减少拷贝
    buffer_a: wgpu::Buffer,
    buffer_b: wgpu::Buffer,

    // 通用权重缓冲区
    weights_buffer: wgpu::Buffer,
    biases_buffer: wgpu::Buffer,

    // LSTM专用缓冲区
    hidden_states_buffer: wgpu::Buffer,
    cell_states_buffer: wgpu::Buffer,
    dispatch_buffer: wgpu::Buffer,
    initial_hidden_buffer: wgpu::Buffer,
    initial_cell_buffer: wgpu::Buffer,

    // 通用管线
    dense_pipeline: wgpu::ComputePipeline,
    lstm_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,
    residual_pipeline: wgpu::ComputePipeline,
    concat_pipeline: wgpu::ComputePipeline,

    // 不同类型的绑定组布局
    common_bind_group_layout: wgpu::BindGroupLayout,
    lstm_bind_group_layout: wgpu::BindGroupLayout,
    attention_bind_group_layout: wgpu::BindGroupLayout,
    residual_bind_group_layout: wgpu::BindGroupLayout,
    concat_bind_group_layout: wgpu::BindGroupLayout,

    // 性能优化：缓存绑定组
    cached_bind_groups: Option<Vec<wgpu::BindGroup>>,

    // 性能优化：权重是否已上传
    weights_uploaded: bool,

    // 性能优化：预计算的参数数据
    params_data_cache: Vec<NetworkParams>,
}

impl GpuNetworkExecutor {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        // 创建参数缓冲区，使用高性能内存配置
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Network Params Buffer"),
            size: std::mem::size_of::<NetworkParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let max_size = 128 * 128;
        let buffer_size = (max_size * std::mem::size_of::<f32>()) as u64;

        // Ping-Pong缓冲区：减少不必要的内存拷贝
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

        // 创建通用绑定组布局
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

        // 创建LSTM专用绑定组布局 (减少到 8 个绑定)
        // 优化：合并 params+dispatch 为 uniform，合并 initial_hidden+initial_cell 为一个 storage
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

        // 创建Attention专用绑定组布局
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

        // 创建Residual专用绑定组布局
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

        // 创建Concat绑定组布局
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
            buffer_a,
            buffer_b,
            weights_buffer,
            biases_buffer,
            hidden_states_buffer,
            cell_states_buffer,
            dispatch_buffer,
            initial_hidden_buffer,
            initial_cell_buffer,
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
            cached_bind_groups: None,
            weights_uploaded: false,
            params_data_cache: Vec::new(),
        }
    }

    // 性能优化：一次性初始化权重和绑定组
    pub fn initialize_network(&mut self, layers: &[NetworkLayerGPU]) {
        // 上传权重（只需一次）
        if !self.weights_uploaded {
            self.upload_all_weights_and_biases(layers);
            self.weights_uploaded = true;
        }

        // 预计算所有层的参数
        self.params_data_cache.clear();
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
            self.params_data_cache.push(params);
            current_input_size = layer.output_size;
        }

        // 预创建绑定组（使用Ping-Pong策略）
        let bind_groups = self.create_ping_pong_bind_groups(layers);
        self.cached_bind_groups = Some(bind_groups);
    }

    // 执行整个网络的GPU前向传播（极致优化版本）
    pub fn execute_network_forward(&self,
                                   input_data: &[f32],
                                   layers: &[NetworkLayerGPU],
                                   output_size: usize) -> Vec<f32> {

        // 上传输入数据到buffer_a
        self.queue.write_buffer(&self.buffer_a, 0, bytemuck::cast_slice(input_data));

        // 使用缓存的绑定组执行层计算
        if let Some(bind_groups) = &self.cached_bind_groups {
            self.execute_layers_optimized(layers, bind_groups);
        } else {
            // 如果没有缓存，使用原始方法
            let bind_groups = self.create_ping_pong_bind_groups(layers);
            self.execute_layers_optimized(layers, &bind_groups);
        }

        // 最终结果下载（从最后一层的输出buffer）
        let last_buffer = if layers.len() % 2 == 1 { &self.buffer_b } else { &self.buffer_a };
        self.download_result_optimized(last_buffer, output_size)
    }

    // 创建Ping-Pong绑定组
    fn create_ping_pong_bind_groups(&self, layers: &[NetworkLayerGPU]) -> Vec<wgpu::BindGroup> {
        let mut bind_groups = Vec::with_capacity(layers.len());

        for (i, layer) in layers.iter().enumerate() {
            // Ping-Pong策略：奇数层从B读取写入A，偶数层从A读取写入B
            let (input_buf, output_buf) = if i % 2 == 0 {
                (&self.buffer_a, &self.buffer_b)
            } else {
                (&self.buffer_b, &self.buffer_a)
            };

            let layout = match layer.layer_type {
                LayerTypeGPU::Dense => &self.common_bind_group_layout,
                LayerTypeGPU::LSTM => &self.lstm_bind_group_layout,
                LayerTypeGPU::Attention => &self.attention_bind_group_layout,
                LayerTypeGPU::Residual => &self.residual_bind_group_layout,
                LayerTypeGPU::Concat => &self.concat_bind_group_layout,
            };

            let bind_group = match layer.layer_type {
                LayerTypeGPU::Dense => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.params_buffer.as_entire_binding(),
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
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.biases_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Dense Bind Group", i)),
                    })
                },
                LayerTypeGPU::LSTM => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.biases_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.hidden_states_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.cell_states_buffer.as_entire_binding(),
                            },
                            // binding 6: 合并的 uniform 缓冲区 (params + dispatch)
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.params_buffer.as_entire_binding(),
                            },
                            // binding 7: 合并的 storage 缓冲区 (initial_hidden + initial_cell)
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: self.initial_hidden_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} LSTM Bind Group", i)),
                    })
                },
                LayerTypeGPU::Attention => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.weights_buffer.as_entire_binding(),
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
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.biases_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: input_buf.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Residual Bind Group", i)),
                    })
                },
                LayerTypeGPU::Concat => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.params_buffer.as_entire_binding(),
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

    // 优化的层执行：减少CPU-GPU同步，批量更新参数
    fn execute_layers_optimized(&self, layers: &[NetworkLayerGPU], bind_groups: &[wgpu::BindGroup]) {
        // 首先准备所有层的参数
        let mut all_params: Vec<NetworkParams> = Vec::with_capacity(layers.len());
        for (i, layer) in layers.iter().enumerate() {
            let params = if i < self.params_data_cache.len() {
                self.params_data_cache[i]
            } else {
                // 回退方案
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

        // 创建单个命令编码器和一个compute pass来执行所有层
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Optimized Compute Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Optimized Compute Pass"),
                timestamp_writes: None,
            });

            for (i, layer) in layers.iter().enumerate() {
                // 参数已在pass外准备好，直接写入
                self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&all_params[i]));

                // 选择管线
                let pipeline = match layer.layer_type {
                    LayerTypeGPU::Dense => &self.dense_pipeline,
                    LayerTypeGPU::LSTM => &self.lstm_pipeline,
                    LayerTypeGPU::Attention => &self.attention_pipeline,
                    LayerTypeGPU::Residual => &self.residual_pipeline,
                    LayerTypeGPU::Concat => &self.concat_pipeline,
                };

                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_groups[i], &[]);

                // 优化的工作组调度
                let (x, y, z) = Self::calculate_optimal_dispatch(layer.output_size as u32);
                cpass.dispatch_workgroups(x, y, z);
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    // 计算最优的工作组调度
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

    fn upload_all_weights_and_biases(&self, layers: &[NetworkLayerGPU]) {
        let mut all_weights = Vec::new();
        let mut all_biases = Vec::new();

        for layer in layers {
            all_weights.extend_from_slice(&layer.weights_flattened);
            all_biases.extend_from_slice(&layer.biases);
        }

        if !all_weights.is_empty() {
            self.queue.write_buffer(&self.weights_buffer, 0, bytemuck::cast_slice(&all_weights));
        }

        if !all_biases.is_empty() {
            self.queue.write_buffer(&self.biases_buffer, 0, bytemuck::cast_slice(&all_biases));
        }
    }

    // 优化的结果下载：减少等待时间
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

// 为GPU优化创建简化的层结构
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

// 管线创建辅助函数
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