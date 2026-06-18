import os
import csv
import itertools
import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n⚠ [오류] matplotlib 라이브러리가 설치되어 있지 않습니다.")
    print("그래프 시각화를 위해 설치가 필요합니다: pip install matplotlib\n")
    plt = None

# Import custom modules
from pet_env import PetSim
from dqn_agent import DQNAgent

# Enable Korean font on Windows for matplotlib (if imported)
if plt is not None:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

def main():
    # Initialize Agent
    agent = DQNAgent(
        state_size=14,
        action_size=9,
    )
    
    # Load model
    model_loaded = agent.load_model('pet_dqn_model.pth')
    if model_loaded:
        print("★ 학습된 모델 가중치(pet_dqn_model.pth)를 성공적으로 로드했습니다. ★")
    else:
        print("⚠ 학습된 모델('pet_dqn_model.pth')을 찾지 못했습니다. 초기 무작위 가중치 상태로 진행합니다.")
    
    # Force epsilon to 0 to evaluate the learned policy
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    # Personality traits
    traits = ["활발함(Active)", "먹성(Gluttony)", "인내심(Patience)", "호기심(Curiosity)", "충성도(Loyalty)"]
    action_names = [
        "대기 (Idle)", "배회 (Wander)", "식사 (Eat)", 
        "침대 수면 (Sleep)", "바닥 수면 (Floor)", "놀이 (Play)", 
        "그루밍 (Groom)", "목욕 (Wash)", "정리 (Clean)"
    ]

    # Generate 32 combinations (0 and 1)
    combinations = list(itertools.product([0.0, 1.0], repeat=5))
    
    results = []
    heatmap_data = []
    
    steps_per_run = 1000
    
    print(f"총 {len(combinations)}가지 성격 조합에 대해 각각 {steps_per_run}번의 행동을 시뮬레이션합니다...")
    
    for idx, c in enumerate(combinations):
        # Instantiate environment
        pet = PetSim()
        
        # Reset to clear counts
        pet.reset()
        # Re-set personality after reset because reset() resets personality to 0.5
        pet.set_personality(activeness=c[0], gluttony=c[1], patience=c[2], curiosity=c[3], loyalty=c[4])
        
        # Run simulation
        for step in range(steps_per_run):
            s = pet.get_state_vector()
            
            # Place toy & commands randomly (same logic as main/pretrain to keep environment active)
            if not pet.toy_placed and np.random.rand() < 0.05:
                pet.place_toy()
            if not pet.command_pending and pet.toy_placed and np.random.rand() < 0.03:
                pet.command_pending = True
                
            # Act (pure exploit)
            a = agent.act(s, force_exploit=True)
            pet.step(a)
            
        # Get percentages
        total_steps = pet.total_action_steps
        counts = pet.action_counts
        percentages = [(count / total_steps * 100.0) if total_steps > 0 else 0.0 for count in counts]
        
        results.append({
            "personality": c,
            "percentages": percentages
        })
        heatmap_data.append(percentages)
        
        # Print progress every 8 steps
        if (idx + 1) % 8 == 0 or idx == len(combinations) - 1:
            print(f"진행 상황: {idx + 1}/{len(combinations)}")

    # Write to CSV
    csv_file = "personality_action_analysis.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        header = traits + action_names
        writer.writerow(header)
        for r in results:
            row = list(r["personality"]) + r["percentages"]
            writer.writerow(row)
    print(f"★ CSV 분석 데이터 저장 완료: '{csv_file}'")

    if plt is None:
        print("⚠ matplotlib가 없어 그래프 생성을 건너뜁니다. CSV 파일로 결과를 확인하세요.")
        return

    # Convert to numpy array for plotting
    heatmap_data = np.array(heatmap_data)

    # Plot Heatmap
    fig, ax = plt.subplots(figsize=(15, 13))
    
    # Draw heatmap
    im = ax.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")
    
    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("행동 선택 비율 (%)", rotation=-90, va="bottom", fontsize=12)
    
    # Configure Ticks
    ax.set_xticks(np.arange(len(action_names)))
    ax.set_xticklabels(action_names, fontsize=11, rotation=45, ha="right")
    
    # Generate tick labels for y-axis: e.g. "활1 먹0 인1 호0 충1"
    y_labels = []
    for c in combinations:
        lbl = f"활{int(c[0])} 먹{int(c[1])} 인{int(c[2])} 호{int(c[3])} 충{int(c[4])}"
        y_labels.append(lbl)
        
    ax.set_yticks(np.arange(len(combinations)))
    ax.set_yticklabels(y_labels, fontsize=9)
    
    # Grid lines to separate cells
    ax.set_xticks(np.arange(len(action_names) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(combinations) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Add percentage text inside heatmap cells
    for i in range(len(combinations)):
        for j in range(len(action_names)):
            val = heatmap_data[i, j]
            # Print text only if value > 0.5% to keep it clean
            if val > 0.5:
                color = "white" if val > 45 else "black"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8)
                
    # Labels & Title
    ax.set_title("성격 조합별 행동 선택 비율 분석 (32가지 조합 x 1000걸음 시뮬레이션)", fontsize=16, pad=20, weight='bold')
    ax.set_xlabel("행동 종류", fontsize=12, labelpad=10)
    ax.set_ylabel("성격 조합 (활발함, 먹성, 인내심, 호기심, 충성도)", fontsize=12, labelpad=10)
    
    plt.tight_layout()
    plot_file = "personality_action_analysis.png"
    plt.savefig(plot_file, dpi=200)
    plt.close()
    print(f"★ 시각화 그래프 저장 완료: '{plot_file}'")

if __name__ == "__main__":
    main()
