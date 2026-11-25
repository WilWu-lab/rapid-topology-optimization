clc;
close all;
clear all;

%% 0. [r2 返修补充] 随机种子 + 输出路径 —— 只加种子，不改任何算法/参数
SEED = str2double(getenv('GA_SEED'));
if isnan(SEED), SEED = 1; end
NW = str2double(getenv('GA_WORKERS'));
if isnan(NW) || NW < 1, NW = 20; end
rng(SEED, 'twister');              % 固定全局随机流，保证同一 SEED 可复现
if isempty(gcp('nocreate')), parpool('Processes', NW); end   % 仅并行评估，不改变算法
outDir = fileparts(mfilename('fullpath'));
addpath(outDir);                   % 让同目录的 fun_ahd.m 可解析
t_start = tic;
convFile = fullfile(outDir, sprintf('convergence_seed%d.csv', SEED));
fid_conv = fopen(convFile, 'w');
fprintf(fid_conv, 'generation,best_fitness,best_cost\n');
fclose(fid_conv);

%% 1. 参数设置
pop_size = 500;             
max_a = 6;
min_a = 2;                  
num_genes_s = 6;            
num_genes_beta = max_a * num_genes_s;       
num_genes = 1 + max_a + num_genes_s + num_genes_beta; 
max_generations = 1500;     
mutation_rate = 0.15;       
elite_percentage = 0.20;    

%% 2. 几何与合法区域预处理 (严谨物理映射版)
% --- A. 物理参数定义 ---
L_x = 50;           % 板长宽 50cm           
L_y = 50;
Cut_x = 20;         % 挖掉区域   
Cut_y = 20;

% --- B. 网格参数定义 ---
n_rows = 51; n_cols = 51;    % 从0开始    
dx = L_x / (n_cols - 1);     % 步长1cm
dy = L_y / (n_rows - 1); 
total_points = n_rows * n_cols;     % 网格数目
all_indices = 1:total_points;       % 索引数目 1:2601(51*51)

% --- C. 建立坐标映射 (核心步骤) ---
% 将线性索引 1~2601 转换为行列下标
[r, c] = ind2sub([n_rows, n_cols], all_indices);

% 物理坐标映射表 (1 x 2601 的向量)
% 将行列下标转换为物理坐标 (cm)
% 假设：Row 1 是最上方 (y=50), Row 51 是最下方 (y=0)
% 假设：Col 1 是最左方 (x=0),  Col 51 是最右方 (x=50)
X_coords = (c - 1) * dx;            % 列转X,1->0cm, 51->50cm
Y_coords = (n_rows - r) * dy;       % 行转Y,1->50cm, 51->0cm

% --- D. 定义挖空区域 ---
% 物理条件：挖掉右上角
% 即：X > (50 - 20)  且  Y > (50 - 20)
% 也就是：X > 30cm 且 Y > 30cm 的区域是“空气”
% 注意：边界 X=30 和 Y=30 是板子边缘，必须保留，所以用大于号 
% 但是当x=30时，不符合实验条件，实验中换能器和传感器不应该在板子边缘，因此用大于等于号
is_in_cutout = (X_coords >= (L_x - Cut_x)) & (Y_coords >= (L_y - Cut_y));

% --- E. 获取全局合法索引池 ---
valid_indices_global = all_indices(~is_in_cutout); 

% --- F. 动态定义激励搜索区域 (激励要在上下光纤上)---
% 这里的范围根据你的需求设定，条件：(1<=x<=49,y=45) U (1<=x<=49,y=5)
mask_up = (X_coords >= 1) & (X_coords <= 49) & (abs(Y_coords - 45) < 1e-4) & ~is_in_cutout; 
valid_pool_up = all_indices(mask_up);

mask_down = (X_coords >= 1) & (X_coords <= 49) & (abs(Y_coords - 5) < 1e-4) & ~is_in_cutout; 
valid_pool_down = all_indices(mask_down);

if isempty(valid_pool_up) || isempty(valid_pool_down)
    error('定义的激励物理搜索范围内没有合法的网格点！');
end

%% 3. 初始化种群
pop1 = zeros(pop_size, 1 + max_a); 
pop_s = zeros(pop_size, num_genes_s);

for i = 1:pop_size
    % --- 初始化激励 ---
    a = randi([1, 3]); 
    b = randi([1, 3]);
    num_genes_a = a + b;
    
    while true
        % 随机抽取
        pts_up = valid_pool_up(randi(length(valid_pool_up), 1, a));
        pts_down = valid_pool_down(randi(length(valid_pool_down), 1, b));
        temp_pts = [pts_up, pts_down];
        
        % 【修正】计算真实的物理距离 (欧氏距离)
        % 1. 查表得到 (x,y)
        x_pts = X_coords(temp_pts);
        y_pts = Y_coords(temp_pts);
        coords = [x_pts', y_pts']; % 形成 N x 2 矩阵
        
        % 2. 计算距离矩阵 (需统计工具箱， pdist)
        if length(temp_pts) > 1
            dists = pdist(coords, 'euclidean');
            min_dist = min(dists);
        else
            min_dist = 999; % 只有一个点，距离无限大
        end
        
        % 3. 判断是否满足 5cm 约束
        if min_dist >= 5
            start_col = (max_a - num_genes_a + 2);
            pop1(i, 1) = num_genes_a;
            pop1(i, start_col:end) = temp_pts;
            break;
        end
    end
    
    % --- 初始化接收 (修改版) ---
    current_excitations = pop1(i, (max_a - num_genes_a + 2):end);
    
    % 【新增步骤 1】构建接收点的专用合法池
    % 目的：从全局合法点中，进一步剔除 Y=5 和 Y=45 的点
    % 注意：如果为了追求极致效率，这一步筛选建议放在主循环(for i=1:pop_size)外面只运行一次
    
    % 获取当前全局合法点的 Y 坐标
    global_valid_Ys = Y_coords(valid_indices_global);
    % 获取当前全局合法点的 X 坐标
    global_valid_Xs = X_coords(valid_indices_global);
    
    % 创建掩膜：找出 Y 不等于 5 且 Y 不等于 45且Y不等于0且Y 不等于50 的索引
    % 使用 abs(val - target) > 1e-4 是为了防止浮点数比较的微小误差
    mask_safe_y = (abs(global_valid_Ys - 5) > 1e-4) & (abs(global_valid_Ys - 45) > 1e-4)... 
                & (abs(global_valid_Ys - 0) > 1e-4) & (abs(global_valid_Ys - 50) > 1e-4);

    % 创建掩膜：找出 X 不等于 0 且 X 不等于 50的索引
    mask_safe_x = (abs(global_valid_Xs - 0) > 1e-4) & (abs(global_valid_Xs - 50) > 1e-4);
                

    % 得到接收点专用的合法索引池
    valid_pool_s = valid_indices_global(mask_safe_y & mask_safe_x);
    
    while true
        % 【修改步骤 2】从专用的 valid_pool_s 中抽取，而不是 valid_indices_global
        % 这样生成的点一定满足：在板内 + 不在挖空区 + Y!=5 + Y!=45
        rand_idx = randi(length(valid_pool_s), 1, num_genes_s);
        candidate_s = valid_pool_s(rand_idx);
        
        % 拼接用于冲突检测
        all_points = [current_excitations, candidate_s];
        
        % 【步骤 3】去重检查
        % 1. length(unique(candidate_s)) == num_genes_s 保证接收点互不重叠
        % 2. 这里的 unique 同时也保证了接收点不与激励点重叠
        %    (虽然物理上 Y 坐标已经错开，但保留此逻辑更加严谨)
        if length(unique(all_points)) == (num_genes_a + num_genes_s)
            pop_s(i, :) = candidate_s;
            break; 
        end
    end
end

% --- 初始化 Beta ---
pop_beta = 1.001 + rand(pop_size, num_genes_beta) * (1.03 - 1.001); 
% 合并
population = [pop1, pop_s, pop_beta];

%% 4. 第一代个体适应度 (修正后)
fitness = zeros(pop_size, 1); 

parfor i = 1:pop_size
    % 1. 解码：获取索引
    num_genes_a = population(i, 1);
    % 提取所有相关的点（激励 + 接收）的索引
    index_all = population(i, max_a-num_genes_a+2 : max_a+1+num_genes_s);
    beta_matrix = population(i, max_a+2+num_genes_s:end);
    
    % 2. 【核心修改】查表获取物理坐标 (cm)
    % 直接使用之前定义好的全局映射表 X_coords 和 Y_coords
    x_input = X_coords(index_all); 
    y_input = Y_coords(index_all);
    
    % 3. 代入目标函数
    % 注意：fun1203 内部需要知道哪些是激励，哪些是接收
    % 这里的 x_input 前 num_genes_a 个是激励，后面是接收
    Y = fun_ahd(x_input, y_input, beta_matrix, num_genes_a, num_genes_s);
    fitness(i) = -Y;
end

%% 5. 输出结果优化
[~, best_index] = max(fitness);
best_ind = population(best_index, :);
best_fit = -fitness(best_index);

% 解码最优解的坐标，方便查看
n_a_best = best_ind(1);
idx_a_best = best_ind(max_a - n_a_best + 2 : max_a + 1); % 激励索引
idx_s_best = best_ind(max_a + 2 : max_a + 1 + num_genes_s); % 接收索引

% 查物理坐标
real_xa = X_coords(idx_a_best); real_ya = Y_coords(idx_a_best);
real_xs = X_coords(idx_s_best); real_ys = Y_coords(idx_s_best);

fprintf('=== 最优解信息 ===\n');
fprintf('目标函数值: %.4f\n', best_fit);
fprintf('激励个数: %d\n', n_a_best);
fprintf('激励坐标 (x, y):\n');
disp([real_xa; real_ya]');
fprintf('接收坐标 (x, y):\n');
disp([real_xs; real_ys]');

%% 6. 遗传算法核心迭代 (完整标准版)
% 包含: 锦标赛选择 -> 分段交叉 -> 多策略变异 -> 种群更新 -> 适应度计算
% 确保此前已定义全局变量: valid_pool_up, valid_pool_down, valid_pool_s, X_coords, Y_coords

fprintf('开始遗传算法迭代...\n');
tournament_size = 4; % 锦标赛规模：每次随机选4个，挑最好的做父母

for generation = 1:max_generations
    
    % =====================================================================
    % STEP 1: 精英保留 (Elitism)
    % =====================================================================
    % 目的：将当前种群中适应度最高的个体直接复制到下一代，不经过交叉变异，防止最优解丢失
    num_elites = round(elite_percentage * pop_size); 
    [~, elite_indices] = maxk(fitness, num_elites);
    elites = population(elite_indices, :);
    
    % =====================================================================
    % STEP 2: 生成子代 (Selection & Crossover)
    % =====================================================================
    num_children = pop_size - num_elites; % 需要生成的子代数量
    children_population = zeros(num_children, num_genes); % 预分配内存
    
    for j = 1:num_children
        % --- A. 锦标赛选择父代 (Tournament Selection) ---
        p1_idx = TournamentSelect(fitness, pop_size, tournament_size);
        p2_idx = TournamentSelect(fitness, pop_size, tournament_size);
        parent1 = population(p1_idx, :);
        parent2 = population(p2_idx, :);
        
        % --- B. 交叉操作 (Crossover) ---
        % 我们尝试生成合法的子代，如果生成的子代违反约束（如重叠、距离过近），则重试
        max_attempts = 20; 
        attempt = 0;
        child_success = false;
        
        while attempt < max_attempts
            attempt = attempt + 1;
            child = zeros(1, num_genes);
            
            % [策略 1] 激励部分：整体继承
            % 原因：激励点有严格的结构（右对齐、0填充、特定的行分布），拆分交叉极易破坏结构
            if rand < 0.5, child(1:max_a+1) = parent1(1:max_a+1);
            else,          child(1:max_a+1) = parent2(1:max_a+1); end
            
            % [策略 2] 接收与Beta部分：单点交叉
            % 原因：这部分基因相对独立，可以自由组合以增加多样性
            idx_s_start = max_a + 2;
            cp = randi([idx_s_start, num_genes-1]); % 随机选择切点
            child(idx_s_start:end) = [parent1(idx_s_start:cp), parent2(cp+1:end)];
            
            % [策略 3] 交叉后校验
            % 必须通过所有物理约束（距离、重叠）才能被接受
            if CheckValidity(child, max_a, num_genes_s, X_coords, Y_coords)
                children_population(j, :) = child;
                child_success = true;
                break; % 成功生成，退出重试循环
            end
        end
        
        % 如果重试多次仍失败（极少情况），直接复制父代1，确保种群数量不减少
        if ~child_success
            children_population(j, :) = parent1;
        end
    end
    
    % =====================================================================
    % STEP 3: 变异操作 (Mutation) - 核心逻辑
    % =====================================================================
    % 注意：我们只遍历 children_population，精英(elites)不参与变异
    for k = 1:num_children
        if rand < mutation_rate
            child = children_population(k, :);
            num_a = child(1); % 获取当前个体的激励数量
            
            % --- 变异策略选择 (Roulette Wheel Strategy) ---
            % 20% 概率变异形状因子 Beta
            % 40% 概率变异接收点位置
            % 40% 概率变异激励 (包含位置移动、增加个数、减少个数)
            r_type = rand;
            
            if r_type < 0.2
                % === 类型 A: Beta 变异 ===
                % 在 Beta 基因区随机选一个，微调其数值
                idx = randi([max_a + num_genes_s + 2, num_genes]);
                child(idx) = 1.001 + rand * (1.03 - 1.001);
                
            elseif r_type < 0.6
                % === 类型 B: 接收点位置变异 ===
                s_indices = max_a + 2 : max_a + 1 + num_genes_s;
                mut_idx = s_indices(randi(length(s_indices))); % 选一个要变异的接收点
                
                % [关键] 从 valid_pool_s 中取新点
                % 这个池子已经提前剔除了 Y=5, Y=45 和 挖空区域，保证几何合法
                new_val = valid_pool_s(randi(length(valid_pool_s)));
                
                % 临时替换
                original_val = child(mut_idx);
                child(mut_idx) = new_val;
                
                % 检查是否与其他点（激励或接收）重叠
                all_pts = [child(max_a-num_a+2 : max_a+1), child(s_indices)];
                if length(unique(all_pts)) ~= (num_a + num_genes_s)
                    child(mut_idx) = original_val; % 若重叠，回滚
                end
                
            else
                % === 类型 C: 激励变异 (复杂逻辑) ===
                r_act = rand;
                
                if r_act < 0.5
                    % [C1] 激励位置移动 (Move)
                    % 找到激励基因的有效范围
                    idx_start = max_a - num_a + 2;
                    idx_range = idx_start : max_a + 1;
                    mut_idx = idx_range(randi(length(idx_range)));
                    old_val = child(mut_idx);
                    
                    % 智能判断：如果旧点在上光纤，新点也从上光纤池取，保持分布结构
                    if ismember(old_val, valid_pool_up)
                        new_val = valid_pool_up(randi(length(valid_pool_up)));
                    else
                        new_val = valid_pool_down(randi(length(valid_pool_down)));
                    end
                    
                    % 替换并校验 (需检查5cm间距)
                    child(mut_idx) = new_val;
                    if ~CheckValidity(child, max_a, num_genes_s, X_coords, Y_coords)
                        child(mut_idx) = old_val; % 校验失败则回滚
                    end
                    
                elseif r_act < 0.75 && num_a < max_a
                    % [C2] 增加激励个数 (Add)
                    % 随机决定加在上行还是下行
                    if rand < 0.5
                         new_val = valid_pool_up(randi(length(valid_pool_up)));
                    else
                         new_val = valid_pool_down(randi(length(valid_pool_down)));
                    end
                    
                    % 数据重构：在现有激励点数组后追加新点
                    current_pts = child(max_a - num_a + 2 : max_a + 1);
                    new_pts = [current_pts, new_val]; 
                    
                    % 更新头部计数器 + 重新右对齐填充
                    new_num = num_a + 1;
                    child(1) = new_num;
                    child(2 : max_a + 1) = 0; % 先全部清零
                    child(max_a - new_num + 2 : max_a + 1) = new_pts; % 填入新序列
                    
                    if ~CheckValidity(child, max_a, num_genes_s, X_coords, Y_coords)
                        child = children_population(k, :); % 彻底回滚
                    end
                    
                elseif num_a > min_a
                    % [C3] 减少激励个数 (Delete)
                    idx_range = max_a - num_a + 2 : max_a + 1;
                    del_idx = idx_range(randi(length(idx_range))); % 随机选一个删
                    
                    % 数据重构：剔除选定点
                    current_pts = child(idx_range);
                    current_pts(current_pts == child(del_idx)) = []; 
                    
                    % 更新头部计数器 + 重新右对齐填充
                    new_num = num_a - 1;
                    child(1) = new_num;
                    child(2 : max_a + 1) = 0; 
                    child(max_a - new_num + 2 : max_a + 1) = current_pts;
                    
                    if ~CheckValidity(child, max_a, num_genes_s, X_coords, Y_coords)
                        child = children_population(k, :); % 回滚
                    end
                end
            end
            
            % 将变异后的个体保存
            children_population(k, :) = child;
        end
    end
    
    % =====================================================================
    % STEP 4: 种群合并与更新
    % =====================================================================
    % 新一代 = 精英 (Old) + 子代 (New & Mutated)
    population = [elites; children_population];
    
    % 重新计算全种群适应度 (因为子代是新的)
    parfor i = 1:pop_size
        n_a = population(i, 1);
        idx_all = population(i, max_a-n_a+2 : max_a+1+num_genes_s);
        beta = population(i, max_a+2+num_genes_s:end);
        
        x_in = X_coords(idx_all); 
        y_in = Y_coords(idx_all);
        
        % 计算目标函数 (fun_new 返回越小越好)
        Y_val = fun_ahd(x_in, y_in, beta, n_a, num_genes_s);
        fitness(i) = -Y_val; % 转为越大越好
    end
    
    % 记录历程
    YY(generation) = max(fitness);

    % [r2 返修补充] 每代把当前最优适应度追加进收敛 CSV（逐代落盘，进程中断也留得下已跑过的代）
    fid_conv = fopen(convFile, 'a');
    fprintf(fid_conv, '%d,%.10f,%.10f\n', generation, YY(generation), -YY(generation));
    fclose(fid_conv);
    
    % =====================================================================
    % 【输出模块】每一代输出基础信息，每10代输出完整信息
    % =====================================================================
    % 1. 获取当前代最优个体
    [current_best_fit, current_best_idx] = max(fitness);
    current_best_ind = population(current_best_idx, :);
    
    % 2. 解码通用信息
    current_obj_value = -current_best_fit; % 还原回真实代价
    n_a_curr = current_best_ind(1);        % 激励个数
    
    % 解码坐标 (查表转物理坐标)
    idx_a_curr = current_best_ind(max_a - n_a_curr + 2 : max_a + 1);
    xa_curr = X_coords(idx_a_curr); 
    ya_curr = Y_coords(idx_a_curr);
    
    idx_s_curr = current_best_ind(max_a + 2 : max_a + 1 + num_genes_s);
    xs_curr = X_coords(idx_s_curr); 
    ys_curr = Y_coords(idx_s_curr);
    
    % [r2 返修补充] 每 50 代落一个"当前最优"检查点，长时间跑被中止时仍能评价
    if mod(generation, 50) == 0
        best_individual = current_best_ind;
        best_fitness = current_best_fit;
        best_cost = current_obj_value;
        actuator_coords = [xa_curr(:), ya_curr(:)];
        sensor_coords = [xs_curr(:), ys_curr(:)];
        beta_valid = current_best_ind(max_a + 2 + num_genes_s : max_a + 1 + num_genes_s + n_a_curr * num_genes_s);
        elapsed_seconds = toc(t_start);
        generation_done = generation;
        is_final = false;
        save(fullfile(outDir, sprintf('checkpoint_seed%d.mat', SEED)), ...
             'SEED','best_individual','best_fitness','best_cost', ...
             'actuator_coords','sensor_coords','beta_valid', ...
             'generation_done','is_final','elapsed_seconds');
    end

    % 3. 基础输出 (每一代都执行)
    fprintf('------------------------------------------------------------\n');
    fprintf('Gen: %d | Cost: %.4f | Actuators: %d | Elapsed: %.1f s\n', ...
        generation, current_obj_value, n_a_curr, toc(t_start));
    
    % 为了节省屏幕空间，坐标可以紧凑打印
    % 格式：[x1, y1; x2, y2 ...]
    fprintf('Act_Coords: '); 
    fprintf('[%.1f, %.1f]  ', [xa_curr; ya_curr]); 
    fprintf('\n');
    
    fprintf('Sen_Coords: ');
    fprintf('[%.1f, %.1f]  ', [xs_curr; ys_curr]); 
    fprintf('\n');
    
    % 4. 扩展输出 (每10代额外输出 Beta)
    if mod(generation, 10) == 0
        fprintf('\n>>> [详细信息 - Beta 矩阵 (Generation %d)]:\n', generation);
        beta_curr = current_best_ind(max_a + 2 + num_genes_s : end);
        
        % 格式化打印 Beta，避免一行太长
        % 假设每行打印 6 个数据 (对应 num_genes_s)
        for b_row = 1:n_a_curr
            start_idx = (b_row-1)*num_genes_s + 1;
            end_idx = b_row*num_genes_s;
            % 只有当索引在范围内才打印 (因为 beta 长度是固定的 max_a * num_s，但有效数据只有 n_a * num_s)
            % 这里简单起见打印所有 beta 基因，或者你可以只打印有效部分
            disp(beta_curr(start_idx : min(end_idx, length(beta_curr))));
        end
        fprintf('------------------------------------------------------------\n');
    end
    
    drawnow limitrate;
    
end

%% 7. 结果可视化与输出
% =========================================================================
% A. 解码最优解信息
% =========================================================================
[~, best_idx_final] = max(fitness);
best_ind_final = population(best_idx_final, :);

% 1. 解码激励 (Actuators)
n_a_final = best_ind_final(1);
idx_a_final = best_ind_final(max_a - n_a_final + 2 : max_a + 1);
xa_final = X_coords(idx_a_final); 
ya_final = Y_coords(idx_a_final);

% 2. 解码接收 (Sensors)
idx_s_final = best_ind_final(max_a + 2 : max_a + 1 + num_genes_s);
xs_final = X_coords(idx_s_final); 
ys_final = Y_coords(idx_s_final);

% 3. 提取 Beta 和 目标函数值
best_beta = best_ind_final(max_a + 2 + num_genes_s : end);
best_cost = -fitness(best_idx_final); % 还原为正的代价值

% =========================================================================
% [r2 返修补充] 保存该种子的最优个体与最终指标
% =========================================================================
best_individual = best_ind_final;
best_fitness = -best_cost;                    % 与原代码一致: fitness = -cost
actuator_coords = [xa_final(:), ya_final(:)];
sensor_coords   = [xs_final(:), ys_final(:)];
beta_all   = best_beta(:);                    % 全部 36 个 beta 基因
beta_valid = best_beta(1:n_a_final * num_genes_s);   % 只取有效 (n_a x 6) 段
elapsed_seconds = toc(t_start);
save(fullfile(outDir, sprintf('result_seed%d.mat', SEED)), ...
     'SEED','best_individual','best_fitness','best_cost', ...
     'actuator_coords','sensor_coords','beta_all','beta_valid', ...
     'n_a_final','num_genes_s','pop_size','max_generations', ...
     'mutation_rate','elite_percentage','num_genes', ...
     'YY','elapsed_seconds');

% =========================================================================
% B. 打印最终结果
% =========================================================================
fprintf('\n============================================================\n');
fprintf('                遗传算法优化完成 (Final Result)              \n');
fprintf('============================================================\n');
fprintf('最优目标函数值 (Minimum Cost): %.4f\n', best_cost);
fprintf('激励传感器个数: %d\n', n_a_final);

fprintf('\n>>> 激励坐标 (Actuator) [cm]:\n');
disp([xa_final; ya_final]');

fprintf('>>> 接收坐标 (Sensor) [cm]:\n');
disp([xs_final; ys_final]');

fprintf('>>> 形状因子 Beta (前 %d 个有效):\n', n_a_final * num_genes_s);
% 只打印有效的beta部分 (因为beta矩阵长度是固定的 max_a * num_s)
valid_len = n_a_final * num_genes_s;
disp(best_beta(1:valid_len));
fprintf('============================================================\n');

% =========================================================================
% C. 绘图 1: 算法收敛曲线
% =========================================================================
figure('Color', 'w', 'Name', 'Optimization Convergence');
plot(1:max_generations, -YY, 'LineWidth', 1.5, 'Color', [0.2, 0.4, 0.8]); 
xlabel('Generation (迭代次数)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Cost Function (目标函数值)', 'FontSize', 12, 'FontWeight', 'bold');
title('遗传算法收敛曲线', 'FontSize', 14);
grid on;
box on;
xlim([1, max_generations]);
% 自动调整 ylim 范围，避免被初期极大的惩罚值压缩视图
% 获取后半段数据的最大最小值来设定范围
final_stage_data = -YY(round(max_generations*0.8):end);
if std(final_stage_data) > 1e-6
    ylim([min(final_stage_data)*0.9, max(final_stage_data)*1.1]);
end

% =========================================================================
% D. 绘图 2: 最优布局示意图 (L型板可视化)
% =========================================================================
figure('Color', 'w', 'Name', 'Optimal Sensor Layout');
hold on; axis equal;

% 1. 绘制 L 型板轮廓
% 定义 L 型多边形顶点 (单位 cm)
% (0,0) -> (50,0) -> (50,30) -> (30,30) -> (30,50) -> (0,50) -> (0,0)
plate_x = [0, 50, 50, 30, 30, 0, 0];
plate_y = [0, 0, 30, 30, 50, 50, 0];
fill(plate_x, plate_y, [0.9, 0.9, 0.95], 'EdgeColor', 'k', 'LineWidth', 2);

% 2. 绘制挖空区域示意 (虚线框)
rectangle('Position', [30, 30, 20, 20], 'FaceColor', [1, 0.9, 0.9], ...
          'EdgeColor', 'r', 'LineStyle', '--');
text(40, 40, 'Cutout Area', 'Color', 'r', 'HorizontalAlignment', 'center');

% 3. 绘制关注区域 ROI (中间 30x30cm)
rectangle('Position', [10, 10, 30, 30], 'EdgeColor', [0.5, 0.5, 0.5], ...
          'LineStyle', ':', 'LineWidth', 1.5);
text(12, 12, 'ROI', 'Color', [0.5, 0.5, 0.5]);

% 4. 绘制传感器位置
% 激励点 (红色实心圆)
h_act = scatter(xa_final, ya_final, 120, 'r', 'filled', 'MarkerEdgeColor', 'k');
% 接收点 (蓝色实心方块)
h_sen = scatter(xs_final, ys_final, 100, 'b', 'filled', 's', 'MarkerEdgeColor', 'k');

% 添加标签 (可选：显示坐标数值)
% for i = 1:length(xa_final)
%     text(xa_final(i)+1, ya_final(i), sprintf('A%d', i), 'FontSize', 8);
% end
% for i = 1:length(xs_final)
%     text(xs_final(i)+1, ys_final(i), sprintf('S%d', i), 'FontSize', 8);
% end

% 5. 图例与修饰
legend([h_act, h_sen], {'Actuators (激励)', 'Sensors (接收)'}, ...
       'Location', 'southwest', 'FontSize', 10);
title(['最优传感器布局 (Cost: ' num2str(best_cost, '%.4f') ')'], 'FontSize', 14);
xlabel('X (cm)'); ylabel('Y (cm)');
xlim([-2, 52]); ylim([-2, 52]);
grid on;
set(gca, 'Layer', 'top'); % 确保网格线不被填充遮挡
hold off;


%% --- 辅助函数 ---

% 1. 锦标赛选择
function idx = TournamentSelect(fitness, pop_size, k)
    candidates = randi(pop_size, k, 1);
    [~, best] = max(fitness(candidates));
    idx = candidates(best);
end

% 2. 通用合法性校验 (集成 L 型板物理约束)
function isValid = CheckValidity(child, max_a, num_s, X_coords, Y_coords)
    isValid = true;
    n_a = child(1);
    
    % 解码索引
    idx_a = child(max_a - n_a + 2 : max_a + 1);
    idx_s = child(max_a + 2 : max_a + 1 + num_s);
    
    % 校验 A: 全局重叠检查
    if length(unique([idx_a, idx_s])) ~= (n_a + num_s)
        isValid = false; return;
    end
    
    % 校验 B: 激励点 5cm 物理距离约束
    if n_a >= 2
        xa = X_coords(idx_a); ya = Y_coords(idx_a);
        % pdist 计算所有点对的欧氏距离
        if min(pdist([xa', ya'], 'euclidean')) < 5
            isValid = false; return;
        end
    end
end
