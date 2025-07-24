"""
【概要】
入力画像と相似変換によって変換した出力画像から回転角度θとスケールパラメータsをガウス・ニュートン法によって推定するプログラム
【使用方法】
入力：
・元画像
・相似変換した画像
出力：
・回転角度
・スケールパラメータ
実行：
python gauss_newton_method.py input/color/Lenna.bmp transformed_image.jpg
【情報】
作成者：勝田尚樹
作成日：2025/7/23
"""
import sys
import cv2
import numpy as np

# x方向とy方向に平滑微分フィルタを適用する
def apply_smoothing_differrential_filter(img, kernel_size=3, sigma=1):
    # 平滑化
    img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
    cv2.imshow("img_blurred", img_blurred)
    cv2.waitKey(0)
    # 微分
    # 単純な差分フィルタ
    kernel_dx = np.array([[-1, 0, 1]], dtype=np.float32)
    kernel_dy = np.array([[-1], [0], [1]], dtype=np.float32)
    # フィルタ適用
    dx = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dx)
    dy = cv2.filter2D(img_blurred, cv2.CV_64F, kernel_dy)
    # 表示用に変換
    dx_disp = cv2.convertScaleAbs(dx)
    dy_disp = cv2.convertScaleAbs(dy)
    cv2.imshow("dx", dx_disp)
    cv2.imshow("dy", dy_disp)
    cv2.waitKey(0)
    return dx_disp, dy_disp

# ガウスニュートン法によりパラメータを推定する
def estimate_by_gauss_newton_method(I, I_prime, I_prime_dx, I_prime_dy):
    # 初期値設定
    theta = np.deg2rad(45)
    scale = 2
    threshold = 1e-6
    max_loop = 1000

    H, W = I.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for i in range(max_loop):
        # JθとJθθの計算
        dxprime_dtheta = -scale * (x_coords * np.sin(theta) + y_coords * np.cos(theta))
        dyprime_dtheta = scale * (x_coords * np.cos(theta) - y_coords * np.sin(theta))
        J_theta_mat = (I_prime - I) * (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta)
        J_theta = np.sum(J_theta_mat)
        J_theta_theta_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) ** 2 
        J_theta_theta = np.sum(J_theta_theta_mat)
        # JSとJSSの計算
        dxprime_dscale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        dyprime_dscale = x_coords * np.sin(theta) + y_coords * np.cos(theta)
        J_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_scale = np.sum(J_scale_mat)
        J_scale_scale_mat = (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale) ** 2 
        J_scale_scale = np.sum(J_scale_scale_mat)
        # JθSの計算
        # dxprime_dthetascale = - x_coords * np.sin(theta) - y_coords * np.cos(theta)
        # dyprime_dthetascale = x_coords * np.cos(theta) - y_coords * np.sin(theta)
        # J_theta_scale_mat = (I_prime - I) * (I_prime_dx * dxprime_dthetascale + I_prime_dy * dyprime_dthetascale)
        J_theta_scale_mat = (I_prime_dx * dxprime_dtheta + I_prime_dy * dyprime_dtheta) * (I_prime_dx * dxprime_dscale + I_prime_dy * dyprime_dscale)
        J_theta_scale = np.sum(J_theta_scale_mat)

        nabla_u_J = np.array([J_theta, J_scale])
        H_u = np.array([[J_theta_theta, J_theta_scale],
                        [J_theta_scale, J_scale_scale]])
        H_u_inv = np.linalg.inv(H_u)
        delta_theta, delta_scale =  - H_u_inv @ nabla_u_J
        # print(delta_theta, delta_scale)
        if np.abs(delta_theta) < threshold and np.abs(delta_scale) < threshold:
            break
        theta += delta_theta
        scale += delta_scale
    print(f"反復回数：{i}")
    # breakpoint()
    return theta, scale

def main():
    # データ準備
    if len(sys.argv) != 3:
        print("Usage: python gauss_newton_method.py {元画像のパス} {相似変換した画像のパス}")
        sys.exit(1)
    img_input_path = sys.argv[1]
    img_output_path = sys.argv[2]
    img_input = cv2.imread(img_input_path, cv2.IMREAD_GRAYSCALE)
    img_output = cv2.imread(img_output_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("input",img_input)
    cv2.imshow("output", img_output)
    cv2.waitKey(0)
    # 平滑微分フィルタを適用
    img_output_dx, img_output_dy = apply_smoothing_differrential_filter(img_output, kernel_size=5, sigma=2)
    # ガウスニュートン法によりパラメータを推定
    theta, scale = estimate_by_gauss_newton_method(img_input, img_output, img_output_dx, img_output_dy)
    print(f"(deg):{np.rad2deg(theta)},\t (rad):{theta},\t (scale):{scale}")
    # 可視化

if __name__ == "__main__":
    main()