#ifndef RTDVSR_INFERENCER_HPP
#define RTDVSR_INFERENCER_HPP

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvInferPlugin.h"
#include <NvInfer.h>

class RTDVSRInferencer {
public:
    RTDVSRInferencer(const std::string& enginePath, int widthLR, int heightLR, int scaleFactor = 2, int hiddenDim = 64);
    ~RTDVSRInferencer();

    bool processFrame(const cv::Mat& inputLR, cv::Mat& outputSR);

    void resetState();

private:
    void loadEngine(const std::string& enginePath);
    void allocateBuffers();
    void freeBuffers();

    int m_widthLR;
    int m_heightLR;
    int m_widthSR;
    int m_heightSR;
    int m_scaleFactor;
    int m_hiddenDim;

    size_t m_inputSizeBytes;
    size_t m_outputSizeBytes;
    size_t m_stateSizeBytes;

    nvinfer1::IRuntime* m_runtime{nullptr};
    nvinfer1::ICudaEngine* m_engine{nullptr};
    nvinfer1::IExecutionContext* m_context{nullptr};
    cudaStream_t m_stream{nullptr};

    float* d_inputLR{nullptr};
    float* d_outputSR{nullptr};
    float* d_stateA{nullptr};
    float* d_stateB{nullptr};

    bool m_pingPong{false};
};

#endif // RTDVSR_INFERENCER_HPP