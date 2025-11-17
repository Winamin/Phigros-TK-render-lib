use std::sync::Arc;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

// 为GPU实现创建统一的缓冲区管理
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct NetworkParams {
    pub input_size: u32,
    pub output_size: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    pub layer_index: u32,  // 用于标识当前层
    pub extra_params: u32, // 通用额外参数
}

// 统一的GPU网络执行器
#[derive(Debug)]
pub struct GpuNetworkExecutor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    
    // 统一的参数缓冲区
    params_buffer: wgpu::Buffer,
    
    // 通用的输入/输出缓冲区
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    
    // 通用权重缓冲区
    weights_buffer: wgpu::Buffer,
    biases_buffer: wgpu::Buffer,
    
    // LSTM专用缓冲区
    hidden_states_buffer: wgpu::Buffer,
    cell_states_buffer: wgpu::Buffer,
    dispatch_buffer: wgpu::Buffer,
    initial_hidden_buffer: wgpu::Buffer,
    initial_cell_buffer: wgpu::Buffer,
    
    // 各层类型专用缓冲区
    //dense_layer_buffer: wgpu::Buffer,
    //lstm_layer_buffer: wgpu::Buffer,
    //attention_layer_buffer: wgpu::Buffer,
    //residual_layer_buffer: wgpu::Buffer,
    
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
        
        // 创建适当大小的缓冲区以支持高性能计算（16M floats = 64MB）
        let max_size = 16 * 1024 * 1024;
        // 使用高性能缓冲区配置
        let buffer_size = (max_size * std::mem::size_of::<f32>()) as u64;
        
        let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Input Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Output Buffer"),
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
            contents: bytemuck::cast_slice(&[0u32; 2]), // [t, direction]
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
        
        // 创建各层类型专用128MB缓冲区
        let layer_buffer_size = 128 * 1024 * 1024; // 128MB
        
        let dense_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dense Layer 128MB Buffer"),
            size: layer_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let lstm_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LSTM Layer 128MB Buffer"),
            size: layer_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let attention_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Attention Layer 128MB Buffer"),
            size: layer_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let residual_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Residual Layer 128MB Buffer"),
            size: layer_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
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
        
        // 创建LSTM专用绑定组布局，匹配LSTM着色器的需求
        let lstm_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // binding 0: input (storage, read)
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
                // binding 1: weights (storage, read)
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
                // binding 2: output (storage, read_write)
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
                // binding 3: biases (storage, read)
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
                // binding 4: hidden_states (storage, read_write)
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
                // binding 5: cell_states (storage, read_write)
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
                // binding 6: params (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 7: dispatch (storage, read)
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
                // binding 8: initial_hidden (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 9: initial_cell (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
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
        
        // 创建Attention专用绑定组布局，匹配Attention着色器的需求
        let attention_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // binding 0: params (storage, read)
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
                // binding 1: input (storage, read)
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
                // binding 2: q_weights (storage, read)
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
                // binding 3: k_weights (storage, read)
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
                // binding 4: v_weights (storage, read)
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
                // binding 5: output (storage, read_write)
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
        
        // 创建Residual专用绑定组布局，匹配Residual着色器的需求
        let residual_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // binding 0: input (storage, read)
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
                // binding 1: weights (storage, read)
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
                // binding 2: output (storage, read_write)
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
                // binding 3: biases (storage, read)
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
                // binding 4: residual_input (storage, read)
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
                // binding 5: params (storage, read)
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
            input_buffer,
            output_buffer,
            weights_buffer,
            biases_buffer,
            hidden_states_buffer,
            cell_states_buffer,
            dispatch_buffer,
            initial_hidden_buffer,
            initial_cell_buffer,
           // dense_layer_buffer,
            //lstm_layer_buffer,
            //attention_layer_buffer,
            //residual_layer_buffer,
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
        }
    }
    
    // 执行整个网络的GPU前向传播（最小化CPU-GPU交换）
    pub fn execute_network_forward(&self, 
        input_data: &[f32], 
        layers: &[NetworkLayerGPU], 
        output_size: usize) -> Vec<f32> {
        
        // 性能监控开始
        let start_time = std::time::Instant::now();
        let upload_start = std::time::Instant::now();
        
        // 一次性将所有输入数据上传到GPU
        self.queue.write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(input_data));
        
        // 一次性上传所有层的权重和偏置
        self.upload_all_weights_and_biases(layers);
        
        let _upload_duration = upload_start.elapsed();
        let compute_start = std::time::Instant::now();
        
        // 预创建所有层的绑定组，避免频繁创建
        let bind_groups = self.create_bind_groups_once(layers);
        
        // 合并所有层计算到单个命令编码器中
        self.execute_layers_efficiently(layers, &bind_groups);
        
        let _compute_duration = compute_start.elapsed();
        let download_start = std::time::Instant::now();
        
        // 最终结果从GPU下载
        let final_result = self.download_result(output_size);
        
        let _download_duration = download_start.elapsed();
        //let total_duration = start_time.elapsed();
        
        // 打印性能统计信息
        //eprintln!("GPU计算完成 - 总耗时: {:?}", total_duration);
        
        final_result
    }
    
    // 预创建所有层的绑定组
    fn create_bind_groups_once(&self, layers: &[NetworkLayerGPU]) -> Vec<wgpu::BindGroup> {
        let mut bind_groups = Vec::with_capacity(layers.len());
        
        for (i, layer) in layers.iter().enumerate() {
            // 根据层类型选择正确的绑定组布局
            let layout = match layer.layer_type {
                LayerTypeGPU::Dense => &self.common_bind_group_layout,
                LayerTypeGPU::LSTM => &self.lstm_bind_group_layout,
                LayerTypeGPU::Attention => &self.attention_bind_group_layout,
                LayerTypeGPU::Residual => &self.residual_bind_group_layout,
                LayerTypeGPU::Concat => &self.concat_bind_group_layout,
            };
            
            // 创建绑定组
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
                                resource: self.input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.output_buffer.as_entire_binding(),
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
                        label: Some(&format!("Layer {} Reused Bind Group", i)),
                    })
                },
                LayerTypeGPU::LSTM => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.output_buffer.as_entire_binding(),
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
                            wgpu::BindGroupEntry {
                                binding: 6,
                                resource: self.params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 7,
                                resource: self.dispatch_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 8,
                                resource: self.initial_hidden_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 9,
                                resource: self.initial_cell_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} LSTM Reused Bind Group", i)),
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
                                resource: self.input_buffer.as_entire_binding(),
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
                                resource: self.output_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Attention Reused Bind Group", i)),
                    })
                },
                LayerTypeGPU::Residual => {
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: self.input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.weights_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: self.output_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: self.biases_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 4,
                                resource: self.input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.params_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Residual Reused Bind Group", i)),
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
                                resource: self.input_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 5,
                                resource: self.output_buffer.as_entire_binding(),
                            },
                        ],
                        label: Some(&format!("Layer {} Concat Reused Bind Group", i)),
                    })
                },
            };
            
            bind_groups.push(bind_group);
        }
        
        bind_groups
    }
    
    // 使用单个命令编码器高效执行所有层
    fn execute_layers_efficiently(&self, layers: &[NetworkLayerGPU], bind_groups: &[wgpu::BindGroup]) {
        // 创建单个高性能编码器用于所有层
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Optimized Multi-Layer Compute Encoder"),
        });
        
        let mut current_output_size = 0usize;
        
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Optimized Multi-Layer Compute Pass"),
                timestamp_writes: None,
            });
            
            for (i, layer) in layers.iter().enumerate() {
                // 根据层类型选择正确的管线
                let pipeline = match layer.layer_type {
                    LayerTypeGPU::Dense => &self.dense_pipeline,
                    LayerTypeGPU::LSTM => &self.lstm_pipeline,
                    LayerTypeGPU::Attention => &self.attention_pipeline,
                    LayerTypeGPU::Residual => &self.residual_pipeline,
                    LayerTypeGPU::Concat => &self.concat_pipeline,
                };
                
                // 更新参数缓冲区
                let params = NetworkParams {
                    input_size: current_output_size as u32,
                    output_size: layer.output_size as u32,
                    batch_size: 1,
                    seq_len: layer.seq_len as u32,
                    layer_index: i as u32,
                    extra_params: match layer.layer_type {
                        LayerTypeGPU::LSTM => (1024u32 << 16) | 0u32,
                        LayerTypeGPU::Dense | LayerTypeGPU::Attention | LayerTypeGPU::Residual => {
                            (1u32 << 8) | 1u32
                        }
                        LayerTypeGPU::Concat => layer.num_inputs as u32, // 对于Concat层，extra_params存储输入数量
                    },
                };
                
                self.queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
                
                // 设置管线和绑定组
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &bind_groups[i], &[]);
                
                // 计算工作组调度 - 最大化工作组数量
                let optimal_workgroup_size = 256;
                let total_work_items = (layer.output_size as u32 + optimal_workgroup_size - 1) / optimal_workgroup_size;
                
                // 使用3D网格最大化并行度
                // WebGPU限制：每个维度最大65535，总工作组数理论无上限（实际受GPU资源限制）
                let max_dim = 65535u32;
                
                if total_work_items > max_dim * max_dim {
                    // 超大任务：使用完整3D网格
                    let cube_root = (total_work_items as f64).powf(1.0 / 3.0) as u32;
                    let x_count = cube_root.min(max_dim);
                    let y_count = cube_root.min(max_dim);
                    let z_count = ((total_work_items + x_count * y_count - 1) / (x_count * y_count)).min(max_dim);
                    cpass.dispatch_workgroups(x_count, y_count, z_count);
                } else if total_work_items > max_dim {
                    // 大任务：使用2D网格
                    let sqrt_val = (total_work_items as f64).sqrt() as u32;
                    let x_count = sqrt_val.min(max_dim);
                    let y_count = ((total_work_items + x_count - 1) / x_count).min(max_dim);
                    cpass.dispatch_workgroups(x_count, y_count, 1);
                } else {
                    // 小任务：使用1D网格
                    cpass.dispatch_workgroups(total_work_items, 1, 1);
                }
                
                current_output_size = layer.output_size;
            }
        }
        
        // 只提交一次，减少CPU-GPU交互
        self.queue.submit(Some(encoder.finish()));
    }
    
    fn upload_all_weights_and_biases(&self, layers: &[NetworkLayerGPU]) {
        // 将所有层的权重和偏置合并并上传到GPU
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
    
    fn download_result(&self, output_size: usize) -> Vec<f32> {
        // 创建高性能临时缓冲区以下载结果
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("High Performance Result Buffer"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // 使用高性能编码器
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("High Performance Download Encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &result_buffer, 0, 
            (output_size * std::mem::size_of::<f32>()) as u64);
        
        // 提交并等待GPU完成
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
        
        // 等待GPU完成所有操作
        let _ = self.device.poll(wgpu::PollType::Wait);
        
        // 使用简单的同步方式
        let buffer_slice = result_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        
        // 再次等待确保完成
        let _ = self.device.poll(wgpu::PollType::Wait);
        
        // 高效读取数据
        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        
        // 清理资源
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
    pub num_inputs: usize, // 用于Concat层，表示输入数量
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
        source: wgpu::ShaderSource::Wgsl(include_str!("core/shaders/network_compute.wgsl").into()),
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
        source: wgpu::ShaderSource::Wgsl(include_str!("lstm.wgsl").into()),
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
        source: wgpu::ShaderSource::Wgsl(include_str!("../attention.wgsl").into()),
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
    
    // 重用matmul着色器或创建专门的残差着色器
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("residual"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../residual.wgsl").into()),
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
        source: wgpu::ShaderSource::Wgsl(include_str!("../concat.wgsl").into()),
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