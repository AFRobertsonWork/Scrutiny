#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>

#include "SYSDKInterface.h"


class TOFNode : public rclcpp::Node
{
public:
  TOFNode()
  : Node("scrutiny_tof_node")
  {
    depth8_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(
        "/scrutiny/tof/depth8",
        10);

    depth_raw_pub_ =
      this->create_publisher<sensor_msgs::msg::Image>(
        "/scrutiny/tof/depth_raw",
        10);

    camera_info_pub_ =
      this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "/scrutiny/tof/camera_info",
        10);

    // Visualisation parameters, in millimetres.
    near_min_mm_ =
      this->declare_parameter<int>(
        "near_min_mm",
        300);

    near_max_mm_ =
      this->declare_parameter<int>(
        "near_max_mm",
        900);

    median_blur_enabled_ =
      this->declare_parameter<bool>(
        "median_blur_enabled",
        true);

    median_blur_ksize_ =
      this->declare_parameter<int>(
        "median_blur_ksize",
        3);

    equalize_hist_enabled_ =
      this->declare_parameter<bool>(
        "equalize_hist_enabled",
        false);

    frame_id_ =
      this->declare_parameter<std::string>(
        "frame_id",
        "tof_depth_frame");

    RCLCPP_INFO(
      this->get_logger(),
      "ToF visualisation range: %d mm to %d mm",
      near_min_mm_,
      near_max_mm_);

    initialise_camera();
  }

  ~TOFNode() override
  {
    if (streaming_started_) {
      Synexens::StopStreaming(device_id_);
    }

    if (device_opened_) {
      Synexens::CloseDevice(device_id_);
    }

    if (sdk_initialised_) {
      Synexens::UnInitSDK();
    }
  }

private:
  void initialise_camera()
  {
    auto error = Synexens::InitSDK();

    RCLCPP_INFO(
      this->get_logger(),
      "InitSDK: %d",
      static_cast<int>(error));

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to initialise Synexens SDK");
      return;
    }

    sdk_initialised_ = true;

    int device_count = 0;

    error = Synexens::FindDevice(
      device_count,
      nullptr);

    RCLCPP_INFO(
      this->get_logger(),
      "FindDevice(count): %d, count=%d",
      static_cast<int>(error),
      device_count);

    if (
      error != Synexens::SYERRORCODE_SUCCESS ||
      device_count <= 0)
    {
      RCLCPP_ERROR(
        this->get_logger(),
        "No ToF device found");
      return;
    }

    std::vector<Synexens::SYDeviceInfo> devices(
      device_count);

    error = Synexens::FindDevice(
      device_count,
      devices.data());

    RCLCPP_INFO(
      this->get_logger(),
      "FindDevice(list): %d, count=%d",
      static_cast<int>(error),
      device_count);

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to obtain ToF device list");
      return;
    }

    device_id_ = devices[0].m_nDeviceID;

    error = Synexens::OpenDevice(
      devices[0]);

    RCLCPP_INFO(
      this->get_logger(),
      "OpenDevice: %d",
      static_cast<int>(error));

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to open ToF device");
      return;
    }

    device_opened_ = true;

    error = Synexens::SetFrameResolution(
      device_id_,
      Synexens::SYFRAMETYPE_DEPTH,
      Synexens::SYRESOLUTION_320_240);

    RCLCPP_INFO(
      this->get_logger(),
      "SetFrameResolution: %d",
      static_cast<int>(error));

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to set ToF resolution");
      return;
    }

    read_camera_calibration();

    error = Synexens::StartStreaming(
      device_id_,
      Synexens::SYSTREAMTYPE_DEPTH);

    RCLCPP_INFO(
      this->get_logger(),
      "StartStreaming: %d",
      static_cast<int>(error));

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to start ToF stream");
      return;
    }

    streaming_started_ = true;

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(66),
      std::bind(&TOFNode::tick, this));

    RCLCPP_INFO(
      this->get_logger(),
      "ToF driver ready");

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing /scrutiny/tof/depth8");

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing /scrutiny/tof/depth_raw");

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing /scrutiny/tof/camera_info");
  }

  void read_camera_calibration()
  {
    Synexens::SYIntrinsics intrinsics{};

    const auto error = Synexens::GetIntric(
      device_id_,
      Synexens::SYRESOLUTION_320_240,
      false,
      intrinsics);

    RCLCPP_INFO(
      this->get_logger(),
      "GetIntric(ToF): %d",
      static_cast<int>(error));

    if (error != Synexens::SYERRORCODE_SUCCESS) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Failed to retrieve ToF intrinsics");

      camera_info_available_ = false;
      return;
    }

    camera_info_available_ = true;

    tof_intrinsics_ = intrinsics;

    camera_info_template_.header.frame_id =
      frame_id_;

    camera_info_template_.width =
      static_cast<uint32_t>(
        intrinsics.m_nWidth);

    camera_info_template_.height =
      static_cast<uint32_t>(
        intrinsics.m_nHeight);

    camera_info_template_.distortion_model =
      "plumb_bob";

    camera_info_template_.d.resize(5);

    for (std::size_t i = 0; i < 5; ++i) {
      camera_info_template_.d[i] =
        static_cast<double>(
          intrinsics.m_fltCoeffs[i]);
    }

    const double fx =
      static_cast<double>(
        intrinsics.m_fltFocalDistanceX);

    const double fy =
      static_cast<double>(
        intrinsics.m_fltFocalDistanceY);

    const double cx =
      static_cast<double>(
        intrinsics.m_fltCenterPointX);

    const double cy =
      static_cast<double>(
        intrinsics.m_fltCenterPointY);

    camera_info_template_.k = {
      fx, 0.0, cx,
      0.0, fy, cy,
      0.0, 0.0, 1.0
    };

    camera_info_template_.r = {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0
    };

    camera_info_template_.p = {
      fx, 0.0, cx, 0.0,
      0.0, fy, cy, 0.0,
      0.0, 0.0, 1.0, 0.0
    };

    RCLCPP_INFO(
      this->get_logger(),
      "ToF intrinsics: width=%d height=%d "
      "fx=%.6f fy=%.6f cx=%.6f cy=%.6f",
      intrinsics.m_nWidth,
      intrinsics.m_nHeight,
      intrinsics.m_fltFocalDistanceX,
      intrinsics.m_fltFocalDistanceY,
      intrinsics.m_fltCenterPointX,
      intrinsics.m_fltCenterPointY);

    RCLCPP_INFO(
      this->get_logger(),
      "ToF distortion: "
      "[%.8f, %.8f, %.8f, %.8f, %.8f]",
      intrinsics.m_fltCoeffs[0],
      intrinsics.m_fltCoeffs[1],
      intrinsics.m_fltCoeffs[2],
      intrinsics.m_fltCoeffs[3],
      intrinsics.m_fltCoeffs[4]);

    RCLCPP_INFO(
      this->get_logger(),
      "Synexens ToF-to-RGB rotation:");

    for (int row = 0; row < 3; ++row) {
      RCLCPP_INFO(
        this->get_logger(),
        "[%.8f, %.8f, %.8f]",
        intrinsics.m_fltTOF2RGBRotate[
          row * 3 + 0],
        intrinsics.m_fltTOF2RGBRotate[
          row * 3 + 1],
        intrinsics.m_fltTOF2RGBRotate[
          row * 3 + 2]);
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Synexens ToF-to-RGB translation: "
      "[%.8f, %.8f, %.8f]",
      intrinsics.m_fltTOF2RGBTranslation[0],
      intrinsics.m_fltTOF2RGBTranslation[1],
      intrinsics.m_fltTOF2RGBTranslation[2]);

    RCLCPP_WARN(
      this->get_logger(),
      "The SDK ToF-to-RGB transform is for the "
      "Synexens camera's own RGB sensor, not the ZED camera.");
  }

  void tick()
  {
    if (!streaming_started_) {
      return;
    }

    std::shared_ptr<Synexens::SYFrameData>
      frame_data;

    const auto error =
      Synexens::GetLastFrameDataSafety(
        device_id_,
        frame_data);

    if (
      error != Synexens::SYERRORCODE_SUCCESS ||
      !frame_data)
    {
      return;
    }

    struct FrameLocation
    {
      int byte_offset;
      int frame_index;
    };

    std::map<
      Synexens::SYFrameType,
      FrameLocation>
      frame_locations;

    int byte_offset = 0;

    for (
      int index = 0;
      index < frame_data->m_nFrameCount;
      ++index)
    {
      const auto frame_type =
        frame_data->m_pFrameInfo[index]
          .m_frameType;

      frame_locations[frame_type] = {
        byte_offset,
        index
      };

      const int width =
        frame_data->m_pFrameInfo[index]
          .m_nFrameWidth;

      const int height =
        frame_data->m_pFrameInfo[index]
          .m_nFrameHeight;

      if (
        frame_type ==
        Synexens::SYFRAMETYPE_DEPTH)
      {
        byte_offset +=
          width * height * 2;
      }
      else {
        byte_offset +=
          width * height;
      }
    }

    const auto depth_location =
      frame_locations.find(
        Synexens::SYFRAMETYPE_DEPTH);

    if (
      depth_location ==
      frame_locations.end())
    {
      return;
    }

    const int frame_index =
      depth_location->second.frame_index;

    const int data_offset =
      depth_location->second.byte_offset;

    const int width =
      frame_data->m_pFrameInfo[frame_index]
        .m_nFrameWidth;

    const int height =
      frame_data->m_pFrameInfo[frame_index]
        .m_nFrameHeight;

    auto *depth_pointer =
      reinterpret_cast<unsigned short *>(
        reinterpret_cast<unsigned char *>(
          frame_data->m_pData) +
        data_offset);

    cv::Mat depth16(
      height,
      width,
      CV_16UC1,
      depth_pointer);

    // Clone because the SDK owns the original memory.
    const cv::Mat depth16_copy =
      depth16.clone();

    const std_msgs::msg::Header header =
      create_header();

    publish_raw_depth(
      depth16_copy,
      header);

    publish_camera_info(
      header);

    const cv::Mat depth8 =
      create_visual_depth(
        depth16_copy);

    publish_visual_depth(
      depth8,
      header);
  }

  std_msgs::msg::Header create_header()
  {
    std_msgs::msg::Header header;

    header.stamp =
      this->now();

    header.frame_id =
      frame_id_;

    return header;
  }

  void publish_raw_depth(
    const cv::Mat &depth16,
    const std_msgs::msg::Header &header)
  {
    auto message =
      cv_bridge::CvImage(
        header,
        "16UC1",
        depth16)
      .toImageMsg();

    depth_raw_pub_->publish(
      *message);
  }

  void publish_visual_depth(
    const cv::Mat &depth8,
    const std_msgs::msg::Header &header)
  {
    auto message =
      cv_bridge::CvImage(
        header,
        "mono8",
        depth8)
      .toImageMsg();

    depth8_pub_->publish(
      *message);
  }

  void publish_camera_info(
    const std_msgs::msg::Header &header)
  {
    if (!camera_info_available_) {
      return;
    }

    auto message =
      camera_info_template_;

    message.header =
      header;

    camera_info_pub_->publish(
      message);
  }

  cv::Mat create_visual_depth(
    const cv::Mat &depth16)
  {
    cv::Mat depth8(
      depth16.rows,
      depth16.cols,
      CV_8UC1,
      cv::Scalar(0));

    const float range_mm =
      static_cast<float>(
        near_max_mm_ - near_min_mm_);

    if (range_mm <= 1.0F) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        3000,
        "near_max_mm must be greater "
        "than near_min_mm");

      return depth8;
    }

    for (
      int y = 0;
      y < depth16.rows;
      ++y)
    {
      const auto *source_row =
        depth16.ptr<uint16_t>(y);

      auto *destination_row =
        depth8.ptr<uint8_t>(y);

      for (
        int x = 0;
        x < depth16.cols;
        ++x)
      {
        const uint16_t depth_mm =
          source_row[x];

        if (
          depth_mm == 0 ||
          depth_mm < near_min_mm_ ||
          depth_mm > near_max_mm_)
        {
          destination_row[x] = 0;
          continue;
        }

        const float normalised =
          static_cast<float>(
            near_max_mm_ - depth_mm) /
          range_mm;

        const int pixel_value =
          static_cast<int>(
            normalised * 255.0F);

        destination_row[x] =
          static_cast<uint8_t>(
            std::clamp(
              pixel_value,
              0,
              255));
      }
    }

    if (median_blur_enabled_) {
      int kernel_size =
        median_blur_ksize_;

      if (kernel_size < 3) {
        kernel_size = 3;
      }

      if (kernel_size % 2 == 0) {
        ++kernel_size;
      }

      cv::medianBlur(
        depth8,
        depth8,
        kernel_size);
    }

    if (equalize_hist_enabled_) {
      cv::equalizeHist(
        depth8,
        depth8);
    }

    return depth8;
  }

  rclcpp::Publisher<
    sensor_msgs::msg::Image>::SharedPtr
    depth8_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::Image>::SharedPtr
    depth_raw_pub_;

  rclcpp::Publisher<
    sensor_msgs::msg::CameraInfo>::SharedPtr
    camera_info_pub_;

  rclcpp::TimerBase::SharedPtr timer_;

  sensor_msgs::msg::CameraInfo
    camera_info_template_;

  Synexens::SYIntrinsics
    tof_intrinsics_{};

  unsigned int device_id_{0};

  bool sdk_initialised_{false};
  bool device_opened_{false};
  bool streaming_started_{false};
  bool camera_info_available_{false};

  int near_min_mm_{300};
  int near_max_mm_{900};

  bool median_blur_enabled_{true};
  int median_blur_ksize_{3};
  bool equalize_hist_enabled_{false};

  std::string frame_id_{
    "tof_depth_frame"};
};


int main(
  int argc,
  char **argv)
{
  rclcpp::init(
    argc,
    argv);

  auto node =
    std::make_shared<TOFNode>();

  rclcpp::spin(
    node);

  rclcpp::shutdown();

  return 0;
}