//! 简易保调变速器（SOLA: Synchronous Overlap-Add）。
//!
//! 在解码线程内、resampler 之后、push_samples 之前运行。
//! 通过固定大小的帧 + 重叠区交叉淡入淡出来改变时长，同时保持音调不变。

pub fn normalize_speed(value: f64) -> f32 {
    value.clamp(0.25, 4.0) as f32
}

/// 帧大小（每通道样本数）。1024 在 44.1kHz 下约 23ms，兼顾质量与延迟。
const FRAME_SIZE: usize = 1024;
/// 帧间重叠（每通道样本数）。50% 重叠。
const OVERLAP: usize = FRAME_SIZE / 2;

pub struct TempoProcessor {
    channels: usize,
    input_buffer: Vec<f32>,
    pending_output: Vec<f32>,
    is_first_frame: bool,
    current_speed: f32,
}

impl TempoProcessor {
    pub fn new(channels: usize) -> Self {
        Self {
            channels,
            input_buffer: Vec::new(),
            pending_output: Vec::new(),
            is_first_frame: true,
            current_speed: 1.0,
        }
    }

    /// 处理一段交错的输入样本，产生交错输出。
    /// `speed` > 1.0 加速、< 1.0 减速、≈ 1.0 直通。
    pub fn process(&mut self, input: &[f32], speed: f32, output: &mut Vec<f32>) {
        let bypass = (speed - 1.0).abs() < 0.001;
        let speed_changed = (speed - self.current_speed).abs() > 0.001;

        if bypass || speed_changed {
            // 切换模式/速率时先把内部残留样本冲出去，避免丢样本
            output.extend_from_slice(&self.pending_output);
            self.pending_output.clear();
            self.input_buffer.clear();
            self.is_first_frame = true;
            self.current_speed = speed;
        }

        if bypass {
            output.extend_from_slice(input);
            return;
        }

        self.input_buffer.extend_from_slice(input);

        let frame_total = FRAME_SIZE * self.channels;
        let overlap_total = OVERLAP * self.channels;
        let hop_out = FRAME_SIZE - OVERLAP; // 每帧输出新增样本数（每通道）
        // 每帧输入推进量：speed 越大推进越多（加速）
        let hop_in = ((hop_out as f32 * speed) as usize).max(1) * self.channels;

        let mut input_pos = 0;
        while input_pos + frame_total <= self.input_buffer.len() {
            let frame = &self.input_buffer[input_pos..input_pos + frame_total];

            if self.is_first_frame {
                self.pending_output.extend_from_slice(frame);
                self.is_first_frame = false;
            } else {
                let buf_len = self.pending_output.len();
                if buf_len >= overlap_total {
                    // 等功率交叉淡入淡出（避免线性淡入淡出的响度下降）
                    for i in 0..overlap_total {
                        let theta =
                            (i + 1) as f32 / (overlap_total + 1) as f32 * std::f32::consts::PI / 2.0;
                        let gain_prev = theta.cos();
                        let gain_new = theta.sin();
                        let idx = buf_len - overlap_total + i;
                        self.pending_output[idx] =
                            self.pending_output[idx] * gain_prev + frame[i] * gain_new;
                    }
                } else {
                    // 不应发生（首帧后 pending_output 长度 >= overlap_total），防御性兜底
                    self.pending_output.extend_from_slice(frame);
                }
                // 追加非重叠部分
                self.pending_output.extend_from_slice(&frame[overlap_total..]);
            }

            input_pos += hop_in;
        }

        // 消费已用输入
        if input_pos > 0 {
            self.input_buffer.drain(0..input_pos);
        }

        // 输出所有样本，但保留末尾 overlap_total 个用于下次交叉淡入淡出
        let keep = self.pending_output.len().saturating_sub(overlap_total);
        if keep > 0 {
            output.extend_from_slice(&self.pending_output[..keep]);
            self.pending_output.drain(0..keep);
        }
    }

    /// 在 EOF 时冲出残留样本。会补零成完整帧以处理最后一帧。
    pub fn flush(&mut self, output: &mut Vec<f32>) {
        let frame_total = FRAME_SIZE * self.channels;
        let overlap_total = OVERLAP * self.channels;

        if !self.input_buffer.is_empty() && self.channels > 0 {
            while self.input_buffer.len() < frame_total {
                self.input_buffer.push(0.0);
            }
            let frame = &self.input_buffer[..frame_total];
            if self.is_first_frame {
                output.extend_from_slice(frame);
            } else {
                let buf_len = self.pending_output.len();
                if buf_len >= overlap_total {
                    for i in 0..overlap_total {
                        let theta =
                            (i + 1) as f32 / (overlap_total + 1) as f32 * std::f32::consts::PI / 2.0;
                        let gain_prev = theta.cos();
                        let gain_new = theta.sin();
                        let idx = buf_len - overlap_total + i;
                        self.pending_output[idx] =
                            self.pending_output[idx] * gain_prev + frame[i] * gain_new;
                    }
                }
                self.pending_output.extend_from_slice(&frame[overlap_total..]);
            }
            self.input_buffer.clear();
        }

        output.extend_from_slice(&self.pending_output);
        self.pending_output.clear();
        self.is_first_frame = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 生成正弦波（交错立体声）
    fn sine(samples: usize, freq: f32, sample_rate: u32) -> Vec<f32> {
        (0..samples)
            .flat_map(|i| {
                let v = (2.0 * std::f32::consts::PI * freq * i as f32 / sample_rate as f32).sin();
                [v, v]
            })
            .collect()
    }

    #[test]
    fn bypass_returns_input_unchanged() {
        let mut p = TempoProcessor::new(2);
        let input = sine(2000, 440.0, 44100);
        let mut out = Vec::new();
        p.process(&input, 1.0, &mut out);
        assert_eq!(out.len(), input.len());
        // 内容也一致
        for (a, b) in out.iter().zip(input.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn speed_2x_produces_fewer_samples() {
        let mut p = TempoProcessor::new(2);
        // 足够长的输入让多帧被处理
        let input = sine(44100, 440.0, 44100); // 1 秒
        let mut out = Vec::new();
        p.process(&input, 2.0, &mut out);
        // 2x 加速：输出长度应明显小于输入（接近一半，允许重叠区误差）
        assert!(
            out.len() < input.len() * 9 / 10,
            "expected 2x speed to produce fewer samples, got {} vs {}",
            out.len(),
            input.len()
        );
    }

    #[test]
    fn speed_0_5x_produces_more_samples() {
        let mut p = TempoProcessor::new(2);
        let input = sine(44100, 440.0, 44100);
        let mut out = Vec::new();
        p.process(&input, 0.5, &mut out);
        assert!(
            out.len() > input.len() * 11 / 10,
            "expected 0.5x speed to produce more samples, got {} vs {}",
            out.len(),
            input.len()
        );
    }

    #[test]
    fn flush_emits_remaining_samples() {
        let mut p = TempoProcessor::new(2);
        // 输入不够一帧，process 不产出；flush 后应输出（补零后）
        let input = sine(100, 440.0, 44100);
        let mut out = Vec::new();
        p.process(&input, 2.0, &mut out);
        // 输入 < frame_total，process 不会处理
        assert_eq!(out.len(), 0);
        p.flush(&mut out);
        assert!(out.len() > 0, "flush should emit remaining samples");
    }

    #[test]
    fn speed_change_resets_state() {
        let mut p = TempoProcessor::new(2);
        let input = sine(44100, 440.0, 44100);
        let mut out = Vec::new();
        p.process(&input, 2.0, &mut out);
        out.clear();

        // 切换到 0.5x，应先冲出残留再重新开始
        let input2 = sine(44100, 440.0, 44100);
        p.process(&input2, 0.5, &mut out);
        // 切换时 pending_output 会被冲出，因此切换后立即有输出
        assert!(out.len() > 0);
    }
}
