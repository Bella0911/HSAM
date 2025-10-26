import cv2
import numpy as np
import os
import glob

# --- 配置参数 ---
IMAGE_DIR = "/home/heyan/thyroid/TN3K/masks"
MASK_DIR = "/home/heyan/thyroid/TN3K/scribbles_new"
SUPPORTED_FORMATS = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')

# 显示窗口的大小 (可以任意设置，不影响最终结果)
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1024, 768
LINE_THICKNESS = 3  # 涂鸦线条的粗细

# --- 全局变量 ---
drawing = False
last_point_display = (0, 0)  # 鼠标在显示窗口上的坐标
current_color = (0, 0, 255)  # 默认红色 (B, G, R)
COLORS = {
    '1': (0, 0, 255), '2': (0, 255, 0), '3': (255, 0, 0),
    '4': (0, 255, 255), '5': (255, 0, 255), '6': (255, 255, 0),
    '7': (255, 255, 255)
}


def get_image_files(path):
    image_files = []
    for fmt in SUPPORTED_FORMATS:
        image_files.extend(glob.glob(os.path.join(path, fmt)))
    return sorted(image_files)


# --- 鼠标回调函数 (核心修改) ---
def draw_scribble_pro(event, x, y, flags, param):
    """处理鼠标事件，并将显示坐标映射到原图坐标进行绘制"""
    global drawing, last_point_display

    # 从参数中获取原图尺寸和真实画布
    orig_h, orig_w = param['orig_dims']
    canvas_orig_res = param['canvas_orig_res']

    # 计算坐标映射比例
    x_ratio = orig_w / DISPLAY_WIDTH
    y_ratio = orig_h / DISPLAY_HEIGHT

    # 将当前显示坐标 (x, y) 转换为原图坐标
    orig_x = int(x * x_ratio)
    orig_y = int(y * y_ratio)

    # 将上一个显示坐标转换为原图坐标
    last_point_orig = (int(last_point_display[0] * x_ratio), int(last_point_display[1] * y_ratio))

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point_display = (x, y)
        # 在原图画布上绘制
        cv2.circle(canvas_orig_res, (orig_x, orig_y), LINE_THICKNESS // 2, current_color, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 在原图画布上绘制
            cv2.line(canvas_orig_res, last_point_orig, (orig_x, orig_y), current_color, LINE_THICKNESS, cv2.LINE_AA)
            last_point_display = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# --- 主程序 ---
if __name__ == "__main__":
    os.makedirs(MASK_DIR, exist_ok=True)
    image_paths = get_image_files(IMAGE_DIR)
    if not image_paths:
        print(f"错误：在 '{IMAGE_DIR}' 文件夹中未找到任何图片。")
        exit()

    print("--- 专业版批量涂鸦标记工具 (支持任意分辨率) ---")
    print(f"找到了 {len(image_paths)} 张图片。窗口大小: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print("操作指南与之前相同 (s: 保存并切换, a/d: 切换, c: 清除, 1-7: 换色, ESC: 退出)")

    current_index = 0
    cv2.namedWindow("Pro Annotator")

    while 0 <= current_index < len(image_paths):
        img_path = image_paths[current_index]
        filename = os.path.basename(img_path)
        mask_filename = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(MASK_DIR, mask_filename)

        # 1. 加载原图并获取真实尺寸
        original_image = cv2.imread(img_path)
        orig_h, orig_w, _ = original_image.shape

        # 2. 创建或加载与原图等大的“真实画布”
        if os.path.exists(mask_path):
            canvas_orig_res = cv2.imread(mask_path)
            # 确保加载的mask尺寸与原图一致
            if canvas_orig_res.shape[:2] != (orig_h, orig_w):
                canvas_orig_res = cv2.resize(canvas_orig_res, (orig_w, orig_h))
        else:
            canvas_orig_res = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        # 3. 设置鼠标回调，并传入必要参数
        callback_params = {'orig_dims': (orig_h, orig_w), 'canvas_orig_res': canvas_orig_res}
        cv2.setMouseCallback("Pro Annotator", draw_scribble_pro, callback_params)

        while True:
            # 4. 创建用于显示的缩略图
            display_original = cv2.resize(original_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            display_canvas = cv2.resize(canvas_orig_res, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            display_image = cv2.addWeighted(display_original, 0.6, display_canvas, 0.4, 0)

            progress_text = f"{current_index + 1}/{len(image_paths)}: {filename} ({orig_w}x{orig_h})"
            cv2.putText(display_image, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Pro Annotator", display_image)

            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                current_index = -100
                break
            elif chr(key) in COLORS:
                current_color = COLORS[chr(key)]
            elif key == ord('c'):
                # 清除真实画布
                canvas_orig_res = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                callback_params['canvas_orig_res'] = canvas_orig_res
                print("当前标注已清除。")
            elif key == ord('s'):
                # 5. 保存与原图等大的真实画布
                cv2.imwrite(mask_path, canvas_orig_res)
                print(f"标注已保存至: {mask_path} (尺寸: {orig_w}x{orig_h})")
                current_index += 1
                break
            elif key == ord('d'):
                current_index += 1
                break
            elif key == ord('a'):
                current_index -= 1
                break

    if current_index >= len(image_paths):
        print("\n所有图片已标注完成！")

    cv2.destroyAllWindows()