#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <cstdio>
#include <vector>
#include <queue>
#include <bitset>
#include <climits>

using namespace std;
using namespace std::chrono;

int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}

cv::Mat createOverlay(cv::Size dsize) {
    cv::Mat kernel1d = cv::getGaussianKernel(20, 4, CV_64F);
    cv::Mat kernel2d(20, 20, CV_64F);
    cv::Mat kernel(20, 20, CV_8U);

    double minVal; 
    double maxVal; 
    cv::Point minLoc; 
    cv::Point maxLoc;

    cv::mulTransposed(kernel1d, kernel2d, false);
    cv::minMaxLoc(kernel2d, &minVal, &maxVal, &minLoc, &maxLoc);
    kernel2d.convertTo(kernel, CV_8U, 255.0 / maxVal);
    copyMakeBorder(kernel, kernel, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
    cv::resize(kernel, kernel, dsize, 0, 0, cv::INTER_LINEAR);

    return kernel;
}

class LoLightEncoder {
private:
    int sliceX, sliceY;
    int width, height;
    int frameCount = 0;
    bool flagBit = 0;
    cv::Mat image;
    cv::Mat hilightMask;

    cv::VideoCapture videoCapture;
    int videoTimestamp;

    queue<cv::Mat> frameQueue;
    vector<bool> dataQueue;
    float overlayAlpha;

    int capacity;

public:
    static const int FRAMERATE = 60; 
    static const int BIT_ZERO_FREQ = 30;
    static const int BIT_ONE_FREQ = 20;

    LoLightEncoder(string arg, int sliceX=1, int sliceY=1, float alpha=0.02);
    void feed(const void *ptr, size_t size);
    void feed(const vector<bool> data);
    LoLightEncoder& operator>> (cv::Mat& image);

    cv::Size getSize() {
        return cv::Size(width, height);
    }

    int getQueuedBitLength() {
        return dataQueue.size();
    }

    bool isVideoSource() {
        return videoCapture.isOpened();
    }

    long getTimestamp() {  // in Us
        return frameCount * 1000000L / FRAMERATE;
    }

    inline int dataCapacity() {
        return sliceX * sliceY - 1;
    }
};

LoLightEncoder::LoLightEncoder(string arg, int sliceX, int sliceY, float alpha) {
    image = cv::imread(arg);
    if (image.data == nullptr) {  // maybe a video
        videoCapture.open(arg);
        videoTimestamp = videoCapture.get(cv::CAP_PROP_POS_MSEC) * 1000;
        videoCapture >> image;
    }

    this->sliceX = sliceX;
    this->sliceY = sliceY;
    overlayAlpha = alpha;
    width = image.cols;
    height = image.rows;

    hilightMask = createOverlay(cv::Size(width / sliceX, height / sliceY));
}

void LoLightEncoder::feed(const void *ptr, size_t size) {
    char *p = (char*)ptr;
    for (size_t i=0; i!=size; i++) {
        char byte = p[i];
        for (size_t j=0; j<8*sizeof(char); j++) {
            dataQueue.push_back(byte & 1);
            byte >>= 1;
        }
    }
}

void LoLightEncoder::feed(const vector<bool> data) {
    for (bool bit: data) {
        dataQueue.push_back(bit);
    }
}

LoLightEncoder& LoLightEncoder::operator>> (cv::Mat& result) {
    if (frameQueue.size() > 0) {
        result = frameQueue.front();
        frameQueue.pop();
        frameCount++;
        return *this;
    } else if (getQueuedBitLength() < dataCapacity()) {
        result.release();  // no enough data to encode
        return *this;
    } else {
        static const int DATARATE = gcd(BIT_ONE_FREQ, BIT_ZERO_FREQ);
        static const int FRAME_N = FRAMERATE / DATARATE;
        static const int FRAME_ONE_N = FRAMERATE / BIT_ONE_FREQ;
        static const int FRAME_ZERO_N = FRAMERATE / BIT_ZERO_FREQ;

        int currentTimestamp = getTimestamp();
        // generate new frames
        bool bits[dataCapacity() + 1];

        flagBit = !flagBit;
        bits[0] = flagBit;

        for (int i=0; i<dataCapacity(); i++) {
            bits[i+1] = dataQueue[0];
            dataQueue.erase(dataQueue.begin());
        }

        for (int i=0; i<FRAME_N; i++) {
            cv::Mat frame(height, width, CV_8UC3);
            bool bitOneHilight = i % FRAME_ONE_N < FRAME_ONE_N / 2;
            bool bitZeroHilight = i % FRAME_ZERO_N < FRAME_ZERO_N / 2;
            int positionCount = 0;
            currentTimestamp += 1000000L / FRAMERATE;

            cv::Mat overlay(height, width, CV_8U, cv::Scalar(0));
            for (int yy=0; yy<sliceY; yy++) {
                for (int xx=0; xx<sliceX; xx++) {
                    bool hilight = bits[positionCount++] == 0 ? bitZeroHilight
                                                            : bitOneHilight; 
                    if (hilight) {
                        cv::Rect roi(
                                xx * width / sliceX,
                                yy * height / sliceY,
                                width / sliceX,
                                height / sliceY);
                        hilightMask.copyTo(overlay(roi));
                    }
                }
            }

            // do we need a new frame?
            if (videoCapture.isOpened()) {
                int nextTimestamp;
                bool retVal = true;

                while (true) {
                    nextTimestamp = videoCapture.get(cv::CAP_PROP_POS_MSEC) * 1000;
                    if (abs(nextTimestamp - currentTimestamp) 
                            <= abs(videoTimestamp - currentTimestamp)) {
                        videoTimestamp = nextTimestamp;
                        if (!(retVal = videoCapture.read(image))) {
                            break;
                        }
                        nextTimestamp = videoCapture.get(cv::CAP_PROP_POS_MSEC) * 1000;
                    } else {
                        break;
                    }
                }

                if (retVal == false) {
                    break;
                }
            }

            cvtColor(overlay, overlay, CV_GRAY2BGR);
            addWeighted(image, 1-overlayAlpha, overlay, overlayAlpha, 0.0, frame);
            frameQueue.push(frame);
        }

        return (*this) >> result;
    }
}


/* start pattern: 0b0111111
 * escape: 0b11111 -> 0b111110
 */
vector<bool> encodeString(string str) {
    const static bool START_PATTERN[] = { 0, 1, 1, 1, 1, 1, 1 };
    static char SPECIAL_CHARS[] = { ' ', ',', '.', '\0' };
    vector<bool> result;

    cerr << "String: " << str << endl;

    for (bool bit: START_PATTERN) {
        result.push_back(bit);
        cerr << (int)bit;
    }

    int one_count = 0;
    for (char byte: str) {
        char *mark = strchr(SPECIAL_CHARS, byte);
        if (mark == NULL && !islower(byte)) {
            throw new exception();
        }

        if (mark == NULL) {
            byte -= ('a' - 1);
        } else {
            byte = 28 + (mark - SPECIAL_CHARS);
        }

        for (size_t j=0; j<5; j++) {
            bool bit = byte & 1;
            one_count = bit == 1 ? one_count + 1 : 0;
            result.push_back(bit);
            cerr << (int)bit;

            if (one_count == 5) {
                result.push_back(0);
                one_count = 0;
                cerr << 0;
            }

            byte >>= 1;
        }
    }

    cerr << endl;
    return result;
}

int main(int ac, char** av) {
    const string commandLineDescribe = 
        "{ h help   |      | print help message }"
        "{ @image   |      | image for encoding }"
        "{ @message |      | message to be encoded }"
        "{ n repeat |  0   | repeat message for n times }"
        "{ t time   | 60   | max time (in second) of video }"
        "{ c cols   |  6   | column number of split blocks }"
        "{ r rows   |  4   | row number of split blocks }"
        "{ a alpha  | 0.01 | alpha of the overlay }"
        "{ o output | output.avi | output file name }";
    cv::CommandLineParser parser(ac, av, commandLineDescribe);

    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    struct {
        string imagePath;
        string message;
        int repeat;
        int timeInSecond;
        string outputPath;
        int cols;
        int rows;
        float alpha;
    } arguments;

    arguments.imagePath = parser.get<string>("@image");
    arguments.message = parser.get<string>("@message");
    arguments.repeat = parser.get<int>("repeat");
    arguments.timeInSecond = parser.get<int>("time");
    arguments.cols = parser.get<int>("cols");
    arguments.rows = parser.get<int>("rows");
    arguments.alpha = parser.get<float>("alpha");
    arguments.outputPath = parser.get<string>("output");

    if (arguments.imagePath.empty()) {
        return -1;
    }

    if (arguments.repeat <= 0) {
        arguments.repeat = INT_MAX;
    }

    LoLightEncoder dataEncoder(
            arguments.imagePath,
            arguments.cols,
            arguments.rows,
            arguments.alpha);

    cv::VideoWriter outputVideo(
        arguments.outputPath,
        outputVideo.fourcc('M', 'J', 'P', 'G'),
        LoLightEncoder::FRAMERATE,
        dataEncoder.getSize(),
        true
    );

    if (!outputVideo.isOpened()) {
        cerr << "Could not open the output video for write" << endl;
        return -1;
    }

    while (true) {
        while (dataEncoder.getQueuedBitLength() < dataEncoder.dataCapacity()) {
            dataEncoder.feed(encodeString(arguments.message));
        }

        cv::Mat frame;
        dataEncoder >> frame;

        if (dataEncoder.getTimestamp() > arguments.timeInSecond * 1000000L) {
            break;
        }

        outputVideo << frame;
    }

    outputVideo.release();

    return 0;
}
