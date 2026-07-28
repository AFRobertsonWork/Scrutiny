#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "SYSDKInterface.h"
#include <vector>
#include <map>
#include <chrono>
#include <limits>

class TOFNode : public rclcpp::Node
{
public:
  TOFNode() : Node("scrutiny_tof_node")
  {
    pub_ = this->create_publisher<sensor_msgs::msg::Image>("/scrutiny/tof/depth8", 10);

    // ---- Tunable parameters for near-detail emphasis ----
    // These are in millimetres.
    // Start with something like 300-900 mm and adjust after testing.
    near_min_mm_ = this->declare_parameter<int>("near_min_mm", 300);
    near_max_mm_ = this->declare_parameter<int>("near_max_mm", 900);

    // Noise filtering
    median_blur_enabled_ = this->declare_parameter<bool>("median_blur_enabled", true);
    median_blur_ksize_   = this->declare_parameter<int>("median_blur_ksize", 3);

    // Optional contrast enhancement after scaling
    equalize_hist_enabled_ = this->declare_parameter<bool>("equalize_hist_enabled", false);

    RCLCPP_INFO(this->get_logger(), "TOF near emphasis range: %d mm to %d mm",
                near_min_mm_, near_max_mm_);

    // init SDK
    auto err = Synexens::InitSDK();
    RCLCPP_INFO(this->get_logger(), "InitSDK: %d", static_cast<int>(err));

    // find device
    int dev_count = 0;

    // 1) query count
    err = Synexens::FindDevice(dev_count, nullptr);
    RCLCPP_INFO(this->get_logger(), "FindDevice(count): %d, count=%d", static_cast<int>(err), dev_count);
    if (err != Synexens::SYERRORCODE_SUCCESS || dev_count <= 0) {
      RCLCPP_ERROR(this->get_logger(), "No TOF device found");
      return;
    }

    // 2) fetch list
    std::vector<Synexens::SYDeviceInfo> devs(dev_count);
    err = Synexens::FindDevice(dev_count, devs.data());
    RCLCPP_INFO(this->get_logger(), "FindDevice(list): %d, count=%d", static_cast<int>(err), dev_count);
    if (err != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "FindDevice(list) failed");
      return;
    }

    device_id_ = devs[0].m_nDeviceID;

    // open device
    err = Synexens::OpenDevice(devs[0]);
    RCLCPP_INFO(this->get_logger(), "OpenDevice: %d", static_cast<int>(err));
    if (err != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "OpenDevice failed");
      return;
    }

    // set depth resolution
    Synexens::SetFrameResolution(device_id_, Synexens::SYFRAMETYPE_DEPTH, Synexens::SYRESOLUTION_320_240);

    // start depth stream
    err = Synexens::StartStreaming(device_id_, Synexens::SYSTREAMTYPE_DEPTH);
    RCLCPP_INFO(this->get_logger(), "StartStreaming: %d", static_cast<int>(err));
    if (err != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "StartStreaming failed");
      return;
    }

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(66),   // ~15 FPS
      std::bind(&TOFNode::tick, this));
  }

  ~TOFNode()
  {
    if (device_id_ != 0) {
      Synexens::StopStreaming(device_id_);
      Synexens::CloseDevice(device_id_);
    }
    Synexens::UnInitSDK();
  }

private:
  void tick()
  {
    std::shared_ptr<Synexens::SYFrameData> pFrameData;
    auto err = Synexens::GetLastFrameDataSafety(device_id_, pFrameData);

    if (err != Synexens::SYERRORCODE_SUCCESS || !pFrameData) {
      return;
    }

    // Keep both byte offset and frame index for each frame type
    struct FrameLocation
    {
      int byte_offset;
      int frame_index;
    };

    std::map<Synexens::SYFrameType, FrameLocation> frame_locations;

    int byte_offset = 0;
    for (int i = 0; i < pFrameData->m_nFrameCount; ++i)
    {
      frame_locations[pFrameData->m_pFrameInfo[i].m_frameType] = {byte_offset, i};

      const int w = pFrameData->m_pFrameInfo[i].m_nFrameWidth;
      const int h = pFrameData->m_pFrameInfo[i].m_nFrameHeight;

      // depth frame is 16-bit
      if (pFrameData->m_pFrameInfo[i].m_frameType == Synexens::SYFRAMETYPE_DEPTH) {
        byte_offset += w * h * 2;
      } else {
        byte_offset += w * h;
      }
    }

    auto it = frame_locations.find(Synexens::SYFRAMETYPE_DEPTH);
    if (it == frame_locations.end()) {
      return;
    }

    const int frame_idx = it->second.frame_index;
    const int data_offset = it->second.byte_offset;

    const int w = pFrameData->m_pFrameInfo[frame_idx].m_nFrameWidth;
    const int h = pFrameData->m_pFrameInfo[frame_idx].m_nFrameHeight;

    auto *depth_ptr = reinterpret_cast<unsigned short*>(
      reinterpret_cast<unsigned char*>(pFrameData->m_pData) + data_offset);

    cv::Mat depth16(h, w, CV_16UC1, depth_ptr);

    // Clone so we own the memory safely for processing
    cv::Mat depth16_copy = depth16.clone();

    // Output image showing only the near region with expanded contrast
    cv::Mat depth8(h, w, CV_8UC1, cv::Scalar(0));

    const float range_mm = static_cast<float>(near_max_mm_ - near_min_mm_);
    if (range_mm <= 1.0f) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                           "Invalid near range: near_max_mm must be greater than near_min_mm");
      return;
    }

    for (int y = 0; y < h; ++y)
    {
      const uint16_t* src_row = depth16_copy.ptr<uint16_t>(y);
      uint8_t* dst_row = depth8.ptr<uint8_t>(y);

      for (int x = 0; x < w; ++x)
      {
        const uint16_t d = src_row[x];

        // Ignore invalid pixels and everything outside the near ROI
        if (d == 0 || d < near_min_mm_ || d > near_max_mm_) {
          dst_row[x] = 0;
          continue;
        }

        // Map the near range to full 8-bit contrast.
        // Closer points become brighter.
        const float normalized = static_cast<float>(near_max_mm_ - d) / range_mm;
        const int pixel = static_cast<int>(normalized * 255.0f);

        dst_row[x] = static_cast<uint8_t>(std::clamp(pixel, 0, 255));
      }
    }

    // Light despeckling for noisy ToF close-range data
    if (median_blur_enabled_) {
      int k = median_blur_ksize_;
      if (k < 3) k = 3;
      if (k % 2 == 0) ++k;  // medianBlur requires odd kernel
      cv::medianBlur(depth8, depth8, k);
    }

    // Optional extra local contrast
    if (equalize_hist_enabled_) {
      cv::equalizeHist(depth8, depth8);
    }

    std_msgs::msg::Header header;
    header.stamp = this->now();
    header.frame_id = "tof_depth_frame";

    auto msg = cv_bridge::CvImage(header, "mono8", depth8).toImageMsg();
    pub_->publish(*msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  unsigned int device_id_{0};

  int near_min_mm_{300};
  int near_max_mm_{900};
  bool median_blur_enabled_{true};
  int median_blur_ksize_{3};
  bool equalize_hist_enabled_{false};
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TOFNode>());
  rclcpp::shutdown();
  return 0;
}
