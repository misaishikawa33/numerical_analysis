"""
【概要】
画像を相似変換するプログラム。

【使用方法】
experiment_gauss_newton_method.pyから実行
    # 目的関数を可視化
    # pof.visualize_objective_function(img_input_cropped, img_output_cropped,
    #                                  theta_max=10,
    #                                  theta_min=0,
    #                                  sigma_max=2,
    #                                  simga_min=0.1)
【情報】
作成者：勝田尚樹
作成日：2025/07/23
"""

import numpy as np
import matplotlib.pyplot as plt
import similarity_transform as st

# 目的関数を3次元空間にプロットする。極小値確認用。ただし、めっちゃ実行時間かかる
def visualize_objective_function(img_input, img_output, theta_min=0, theta_max=10, simga_min=0.1, sigma_max=2):
    # パラメータ範囲
    I_prime_org = img_input
    I = img_output
    theta_values = np.arange(theta_min, theta_max, 1)  
    scale_values = np.arange(simga_min, sigma_max, 0.1)    
    # Jの結果格納用 (scale x theta の2次元配列)
    J_values = np.zeros((len(scale_values), len(theta_values)))
    # I と I_prime は事前に用意されているものとする
    for i, scale in enumerate(scale_values):
        for j, theta in enumerate(theta_values):
            # 角度をラジアンに変換
            theta_rad = np.deg2rad(theta)
            # 相似変換を適用
            M = st.compute_M(scale, theta_rad, 0, 0)
            I_prime = st.apply_similarity_transform_reverse(I_prime_org, M)
            I_prime_cropped = st.crop_img_into_circle(I_prime)
            # 目的関数Jを計算
            J = 0.5 * np.sum((I_prime_cropped - I) ** 2)
            J_values[i, j] = J
    # すでに計算済みの J_values, theta_values, scale_values を使用
    Theta, Scale = np.meshgrid(theta_values, scale_values)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    # 3Dサーフェスプロット
    surf = ax.plot_surface(Theta, Scale, J_values, cmap='viridis', edgecolor='none')
    # 軸ラベル
    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Scale')
    ax.set_zlabel('Objective Function J')
    ax.set_title('3D Plot of J(Theta, Scale)')
    # カラーバー
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='J')
    plt.show()