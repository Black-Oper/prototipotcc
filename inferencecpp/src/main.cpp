#include "RTDVSRInferencer.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    std::string videoPath = (argc > 1) ? argv[1] : "test_video.mp4";
    std::string outputDir = (argc > 2) ? argv[2] : "output_frames";
    int maxFrames = (argc > 3) ? std::stoi(argv[3]) : 30;

    const std::string enginePath = "rtdvsr_fp16.engine";
    const int lrWidth = 960;
    const int lrHeight = 540;
    const int scale = 2;

    try {
        std::cout << "Inicializando o motor TensorRT em C++..." << std::endl;
        RTDVSRInferencer inferencer(enginePath, lrWidth, lrHeight, scale);

        cv::VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            std::cerr << "Erro ao abrir o vídeo no caminho: " << videoPath << std::endl;
            return -1;
        }

        fs::create_directories(outputDir);

        cv::Mat rawFrame, inputFrame, outputFrame;
        int frameCount = 0;

        std::cout << "Processando " << maxFrames << " frames do vídeo..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        while (cap.read(rawFrame) && frameCount < maxFrames) {
            if (rawFrame.cols != lrWidth || rawFrame.rows != lrHeight) {
                cv::resize(rawFrame, inputFrame, cv::Size(lrWidth, lrHeight), 0, 0, cv::INTER_CUBIC);
            } else {
                inputFrame = rawFrame;
            }

            if (!inferencer.processFrame(inputFrame, outputFrame)) {
                std::cerr << "Falha na inferência no frame " << frameCount << std::endl;
                return -1;
            }

            char fileName[256];
            snprintf(fileName, sizeof(fileName), "frame_%04d.png", frameCount);
            fs::path outputPath = fs::path(outputDir) / fileName;

            std::vector<uchar> buf;
            if (cv::imencode(".png", outputFrame, buf)) {
                std::ofstream outFile(outputPath, std::ios::binary);
                outFile.write(reinterpret_cast<const char*>(buf.data()), buf.size());
            } else {
                std::cerr << "Erro ao codificar o frame " << frameCount << std::endl;
            }

            frameCount++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        double avgLatency = elapsed.count() / std::max(frameCount, 1);
        double fps = 1000.0 / avgLatency;

        std::cout << "========================================" << std::endl;
        std::cout << "Frames Processados: " << frameCount << std::endl;
        std::cout << "Latência Média:     " << avgLatency << " ms" << std::endl;
        std::cout << "Vazão (FPS):        " << fps << " FPS" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erro fatal: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}