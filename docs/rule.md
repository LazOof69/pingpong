任務預測說明
1. 下一球（第 n 拍）球種預測 (actionId)
• 預測對手在下一次回擊時會使用的球種（例如：殺球、切球、挑球等）。
2.下一球（第 n 拍）落點位置預測 (pointId)
• 預測下一球會落在球場上的哪個具體位置。
3.此回合勝負預測
• 根據此回合（Rally）已進行的擊球序列，預測最終此Rally的發球者是否得分。

資料集介紹
● rally_uid: 所有場次每一分獨立編號
● match: 場次編號
● numberGame: 每一場中的局數編號
● rally_id: 每一局每一分編號
● strikeNumber: 每一分中的拍次編號
● sex: 性別
● gamePlayerId: 此板的主視角的playerid(擊球者)
● gamePlayerOtherId: 此板的主視角的對手的playerid(接球者)
● scoreSelf: 主視角本局得分數
● scoreOther: 主視角對手本局得分數

資料集介紹
● handId: 正反拍
● strengthId: 速度
● spinId: 旋轉
● positionId: 選手初始站位，只在前兩板數值分別表示發球方跟接發球方的站位
● strikeId: 發球 or 接發球 or 第三板之後
● actionId: 球種(預測目標1)
● pointId: 落點，注意這裡會根據接球方是左手持拍或右手持拍，落點數字表示
的位置會不同，因為左右手的正手方向不同(預測目標2)
● serverGetPoint: 發球者是否得分(預測目標3)

Baseline 方法介紹 - 前處理
從train.csv中選取需要之特徵與欄位
對齊訓練資料中最長之資料，不足其長度的用0做尾部補齊(padding)

本競賽旨在評估參賽者利用前 n-1拍的時序資料，對下一拍（第n拍）及當前回合（Rally）結果進行多項預測的綜合能力。為實現對所有參賽者的單一排名，競賽將採用一個綜合評分指標（Overall Score），該指標由以下三項預測任務的個別表現加權平均構成。
各單項任務將採用以下最能反映其特性及應對潛在數據不平衡問題的指標：

任務一：下一拍（第 n 拍）的球種預測（參數名稱：actionId）
評估指標 (S1)：Macro F1-Score
理由：球種類別樣本高度不均衡。Macro F1-Score 給予所有類別相同的權重，以評估模型在各類球種（包括稀有球種）上的平均預測表現。
任務二：下一拍（第 n 拍）的落點 (以九宮格分隔球桌區域) 預測（參數名稱：pointId）
評估指標 (S2)：Macro F1-Score
理由：不同落點區域的擊中頻率不一。Macro F1-Score提供在各區域間均衡的性能評估。
任務三：根據此回合（Rally）已進行的擊球序列，預測最終此Rally的發球者是否得分。 (參數名稱：serverGetPoint)
評估指標 (S3)：AUC-ROC (Area Under the ROC Curve)
理由：AUC-ROC 衡量模型區分兩種結果的整體能力，不受特定分類閾值影響，並且在處理潛在類別不平衡時表現穩健。
所有參賽者的最終排名將依據單一的「綜合評分指標」。該指標的計算方式是將參賽者在三項預測任務中獲得的分數（各單項任務的評估指標分數，均標準化至 0 到 1 範圍），依照預設的權重進行加權平均。
Score=w1*S1+w2*S2+w3*S3

w1,w2,w3訂為：0.4, 0.4 和0.2。這是因為：準確預測對方下一拍的球種 (任務1) 和落點 (任務2) 是戰術分析和應對的核心，具有較高的戰術價值，故給予較高權重；預測整個回合的勝負 (任務3) 是衡量模型對局勢發展和多拍連貫性理解的最終體現，但其重要性可能不及關鍵戰術元素（球種、落點），因此權重相對較低。




資料集簡介
這是一套以「桌球比賽小分（Rally）」為單位所建立的結構化比賽資料集，完整記錄比賽中每一次擊球的狀態、技術動作與比賽脈絡資訊。

檔案說明
檔案名稱 說明
train.csv 訓練資料集，包含完整標註
test.csv 測試資料集，不含預測目標
sample_submission.csv 提交結果格式範例

資料結構與欄位說明
每一筆資料代表 一場比賽中某一小分內的單次揮拍行為，依時間順序排列。

以下是資料集的詳細資訊。 參賽隊伍需要檢視每個ID的參考資料，並選擇適合它們的嵌入方法。 此外，也可以選擇你認為對訓練重要的特徵，並捨棄那些你認為沒用的特徵。

features 說明 Definition

rally_uid 小分的唯一識別碼 Unique ID for each rally

sex 比賽性別（男(1) / 女(2)） Gender category of the match (e.g., male(1)/female(2))

match 比賽的唯一識別碼 Unique ID of the match

numberGame 小局數（第幾局） Game (set) number within the match

rally_id 小局內的小分編號 Rally ID within the game

strikeNumber 小分內的揮拍次序 Stroke number within the rally

scoreSelf 主視角選手的得分 Points won by the main-view player

scoreOther 對側選手的得分 Points won by the opponent player

serverGetPoint 發球者是否得分（1=是, 0=否） Whether the server won the point (1 if yes, 0 if no)

gamePlayerId 主視角選手的 ID ID of the main-view player

gamePlayerOtherId 對側視角選手的 ID ID of the opponent player

strikeId 揮拍狀態或動作類型 Stroke action type or state identifier

handId 正手或反手揮拍 Forehand or backhand stroke indicator

strengthId 擊球力道 Stroke strength level

spinId 球的旋轉方式 Type of spin applied to the ball

pointId 球的落點位置 Landing position of the ball on the table

actionId 擊球方式 Stroke or action type

Player who called a let or timeout

positionId 球員站位區域 Player’s court position

※ serverGetPoint,pointId,actionId為本次競賽預測目標

strikeId
ID 說明 Definition

1 發球 serving

2 接發球 reserve

4 第三板之後 rally

8 無(未錄影) zero

16 暫停 stop

handId
ID 說明 Definition

0 無 zero

1 正拍 forehand

2 反拍 backhand

※ 補充：0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

strengthId
ID 說明 Definition

0 無 zero

1 強 strong

2 中 medium

3 弱 slow

※ 補充：0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

spinId
ID 說明 Definition

0 無 zero

1 上旋 top spin

2 下旋 back spin

3 不旋 no spin

4 側上旋 side top spin

5 側下旋 side back spin

※ 補充：0 代表「無」、「無動作」、「無法判斷狀態」或「其他」

pointId
ID 說明 Definition

0 無 zero

1 正手位置短球 forehand position near net

2 中間短球 middle position near net

3 反手位置短球 backhand position near net

4 正手位置半出台球 forehand position half-long

5 中路半出台球 middle position half-long

6 反手位置短半出台球 backhand position half-long

7 正手位置長球 forehand position long

8 中間長球 middle position long

9 反手位置長球 backhand position long

※ 補充：0代表「無」或「未落在九宮格的位置」（如掛網出界或者直接出界）

actionId
ID 說明 Definition action type

0 無 zero Zero

1 拉球 drive Attack

2 反拉 counter drive Attack

3 殺球 smash Attack

4 擰球 backhand twist Attack

5 快帶 fast drive Attack

6 推擠 fast push Attack

7 挑撥 flip Attack

8 拱球 pimple’s long push Control

9 磕球 pimple’s fast push Control

10 搓球 long push Control

11 擺短 drop shot Control

12 削球 chop Defensive

13 擋球 block Defensive

14 放高球 lob Defensive

15 傳統 traditional Serve

16 勾手 hook Serve

17 逆旋轉 reverse Serve

18 下蹲式 squat Serve

※ 補充：0代表「無」或「其他」（無法判斷之球種）

positionId
ID 說明 Definition

0 無 null

1 左 left

2 中 middle

3 右 right

※ 補充：0 代表「無」、「無動作」、「無法判斷狀態」或「其他」