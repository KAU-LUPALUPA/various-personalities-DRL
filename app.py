import os
import math
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dqn_agent import DQNAgent

app = FastAPI(
    title="Pet Personality DRL AI Server",
    description="DQN 기반 펫 행동 예측 AI API 서버",
    version="1.0.0"
)

# Global Agent Instance
agent = None

ACTION_NAMES = {
    0: "IDLE",
    1: "WANDER",
    2: "EAT",
    3: "SLEEP_BED",
    4: "SLEEP_FLOOR",
    5: "PLAY_TOY",
    6: "GROOM",
    7: "WASH",
    8: "CLEAN_TOY"
}

ACTION_DESCRIPTIONS = {
    0: "제자리에서 대기(IDLE)합니다.",
    1: "방 안을 배회(WANDER)합니다.",
    2: "밥그릇으로 이동하여 식사(EAT)합니다.",
    3: "침대로 이동하여 수면(SLEEP_BED)을 취합니다.",
    4: "제자리 바닥에서 누워 잠(SLEEP_FLOOR)을 청합니다.",
    5: "장난감으로 이동하여 놀기(PLAY_TOY) 행동을 합니다.",
    6: "몸단장(GROOM)을 합니다.",
    7: "씻는 곳으로 이동하여 목욕(WASH)을 합니다.",
    8: "장난감을 상자에 집어넣어 정리(CLEAN_TOY)합니다."
}

# Room coordinates (matching pet_env.py)
ROOM_DIAGONAL = math.sqrt(600 ** 2 + 400 ** 2)  # 721.110255...
FOOD_POS = {"x": 80, "y": 80}
BED_POS = {"x": 520, "y": 80}
WASH_POS = {"x": 80, "y": 320}
TOY_POS = {"x": 520, "y": 320}
CHEST_POS = {"x": 300, "y": 340}

def get_normalized_dist(x: float, y: float, target_x: float, target_y: float) -> float:
    dx = x - target_x
    dy = y - target_y
    dist = math.sqrt(dx ** 2 + dy ** 2)
    return min(1.0, dist / ROOM_DIAGONAL)

@app.on_event("startup")
def startup_event():
    global agent
    # Initialize DQN Agent with matching sizes
    agent = DQNAgent(
        state_size=14,
        action_size=9,
        epsilon=0.0,          # Set epsilon to 0 to only exploit
        epsilon_min=0.0
    )
    
    # Try to load existing model weights
    model_path = "pet_dqn_model.pth"
    if os.path.exists(model_path):
        success = agent.load_model(model_path)
        if success:
            print(f"★ AI Server Startup: Successfully loaded weights from {model_path} ★")
        else:
            print(f"⚠ AI Server Startup: Failed to load weights from {model_path} ⚠")
    else:
        print(f"⚠ AI Server Startup: {model_path} not found. Running with random weights. ⚠")

# Pydantic Schemas
class StateRequest(BaseModel):
    state: list[float] = Field(
        ..., 
        description="14차원의 정규화(0.0~1.0)된 상태 벡터",
        example=[0.8, 0.7, 0.8, 0.2, 0.5, 0.5, 1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5]
    )

class DetailsRequest(BaseModel):
    cleanliness: float = Field(..., ge=0.0, le=1.0, description="청결도 (0.0~1.0)")
    fullness: float = Field(..., ge=0.0, le=1.0, description="포만감 (0.0~1.0)")
    stamina: float = Field(..., ge=0.0, le=1.0, description="체력 (0.0~1.0)")
    boredom: float = Field(..., ge=0.0, le=1.0, description="지루함 (0.0~1.0)")
    pet_x: float = Field(..., description="펫의 X 좌표 (0~600)")
    pet_y: float = Field(..., description="펫의 Y 좌표 (0~400)")
    toy_placed: bool = Field(..., description="장난감이 바닥에 놓여있는지 여부")
    carrying_toy: bool = Field(..., description="펫이 장난감을 입에 물고 있는지 여부")
    command_pending: bool = Field(..., description="정리정돈 대기 명령 여부")
    activeness: float = Field(..., ge=0.0, le=1.0, description="성격: 활동성 (0.0~1.0)")
    gluttony: float = Field(..., ge=0.0, le=1.0, description="성격: 식탐 (0.0~1.0)")
    patience: float = Field(..., ge=0.0, le=1.0, description="성격: 인내심 (0.0~1.0)")
    curiosity: float = Field(..., ge=0.0, le=1.0, description="성격: 호기심 (0.0~1.0)")
    loyalty: float = Field(..., ge=0.0, le=1.0, description="성격: 충성도 (0.0~1.0)")

class PredictionResponse(BaseModel):
    action_id: int = Field(..., description="예측된 행동 번호 (0~8)")
    action_name: str = Field(..., description="행동 영문 명칭")
    description: str = Field(..., description="행동 설명")

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """AI 서버 상태 확인 및 모델 로드 상태 헬스 체크"""
    model_loaded = os.path.exists("pet_dqn_model.pth")
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(agent.device) if agent else "Not Initialized",
        "epsilon": agent.epsilon if agent else 0.0
    }

@app.post("/predict/state", response_model=PredictionResponse)
def predict_by_state(req: StateRequest):
    """14차원 상태 벡터를 통해 행동 예측"""
    if len(req.state) != 14:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="입력 상태 벡터는 반드시 14차원이어야 합니다."
        )
    
    try:
        # Force exploit so that the model selects the highest Q-value action (no exploration)
        action_id = agent.act(req.state, force_exploit=True)
        return PredictionResponse(
            action_id=action_id,
            action_name=ACTION_NAMES[action_id],
            description=ACTION_DESCRIPTIONS[action_id]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"행동 예측 중 오류가 발생했습니다: {str(e)}"
        )

@app.post("/predict/details", response_model=PredictionResponse)
def predict_by_details(req: DetailsRequest):
    """펫의 개별 상세 정보를 받아 정규화 및 거리를 계산한 뒤 행동 예측"""
    try:
        # Calculate distances
        d_food = get_normalized_dist(req.pet_x, req.pet_y, FOOD_POS["x"], FOOD_POS["y"])
        d_bed = get_normalized_dist(req.pet_x, req.pet_y, BED_POS["x"], BED_POS["y"])
        d_wash = get_normalized_dist(req.pet_x, req.pet_y, WASH_POS["x"], WASH_POS["y"])
        
        if req.toy_placed:
            target_toy_pos = CHEST_POS if req.carrying_toy else TOY_POS
            d_toy = get_normalized_dist(req.pet_x, req.pet_y, target_toy_pos["x"], target_toy_pos["y"])
        else:
            d_toy = 1.0  # Default max distance if toy is not placed
            
        # Reconstruct 14-dimensional state vector
        state = [
            req.cleanliness,
            req.fullness,
            req.stamina,
            req.boredom,
            d_food,
            d_bed,
            d_toy,
            d_wash,
            1.0 if req.command_pending else 0.0,
            req.activeness,
            req.gluttony,
            req.patience,
            req.curiosity,
            req.loyalty
        ]
        
        # Predict action
        action_id = agent.act(state, force_exploit=True)
        return PredictionResponse(
            action_id=action_id,
            action_name=ACTION_NAMES[action_id],
            description=ACTION_DESCRIPTIONS[action_id]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"행동 예측 중 오류가 발생했습니다: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Local debugging run
    uvicorn.run(app, host="0.0.0.0", port=8000)
