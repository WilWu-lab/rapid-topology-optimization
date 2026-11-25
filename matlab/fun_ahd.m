function Y=fun_ahd(x,y,beta_matrix,num_genes_a,num_genes_s)
%x和y为激励坐标
%sensor_matrix，每个个体的接收坐标（1*8）
%% 三个惩罚函数, 改用【归一化+AHP加权法】
% 12月3日... (历史注释保留)
% num_genes_a = 5;
% num_genes_s = 6;
% ... (保留原有注释)

%%
% x, y: 输入的物理坐标 (单位: cm)，向量长度 = num_a + num_s
% beta_matrix: 形状因子向量
% num_genes_a: 激励数量
% num_genes_s: 接收数量

%% 1. 数据预处理 (保持不变)
Resolution = 500;
p=1;
distance_matrix = zeros(num_genes_s, num_genes_a); 
canvas = zeros(Resolution, Resolution);    
scale = Resolution / 50; 
% 分离激励和接收坐标
x_a = x(1:num_genes_a) * scale;
y_a = y(1:num_genes_a) * scale;
x_s = x(num_genes_a+1 : num_genes_a+num_genes_s) * scale;
y_s = y(num_genes_a+1 : num_genes_a+num_genes_s) * scale;
points_a = [x_a(:), y_a(:)];
points_s = [x_s(:), y_s(:)];

%% 2. 定义评价区域 (Region of Interest, ROI) (保持不变)
[X_grid, Y_grid] = meshgrid(1:Resolution, 1:Resolution); 
Cutout_X = 30 * scale; 
Cutout_Y = 30 * scale; 
is_plate = ~((X_grid > Cutout_X) & (Y_grid > Cutout_Y));
ROI_Min = 10 * scale; 
ROI_Max = 40 * scale; 
is_center_roi = (X_grid >= ROI_Min & X_grid <= ROI_Max) & ...
                (Y_grid >= ROI_Min & Y_grid <= ROI_Max);
Valid_Mask = is_plate & is_center_roi;

%% 3. RAPID 成像概率场计算 (保持不变)
for i = 1:num_genes_s
    for j = 1:num_genes_a
        sx = points_s(i, 1); sy = points_s(i, 2);
        ax = points_a(j, 1); ay = points_a(j, 2);
        
        num_samples = 50;
        line_x = linspace(sx, ax, num_samples);
        line_y = linspace(sy, ay, num_samples);
        if any(line_x > Cutout_X & line_y > Cutout_Y)
            p = p + 1;
            if p > length(beta_matrix), p = 1; end
            continue; 
        end
        
        D2 = sqrt((sx-ax)^2 + (sy-ay)^2);
        if D2 < 1e-3, continue; end 
        
        distance_matrix(i, j) = D2;
        curr_beta = beta_matrix(p);
        
        Dist_S = sqrt((X_grid - sx).^2 + (Y_grid - sy).^2);
        Dist_A = sqrt((X_grid - ax).^2 + (Y_grid - ay).^2);
        R = (Dist_S + Dist_A) ./ D2;
        
        update_mask = (R < curr_beta);
        
        if any(update_mask(:))
            w = (curr_beta - R(update_mask)) ./ (curr_beta - 1);
            canvas(update_mask) = canvas(update_mask) + w;
        end
        
        p = p + 1;
        if p > length(beta_matrix), p = 1; end
    end
end

%% 4. 计算适应度 (方案一: 归一化 + AHP权重)
% 提取 ROI 内的有效像素值
roi_values = canvas(Valid_Mask);
num_valid_pixels = numel(roi_values);

% --- 指标 1: 无盲区 (Blind Spot) ---
% 原始值计算
Blind_Threshold = 1; 
num_blind = sum(roi_values < Blind_Threshold);
raw_blind_ratio = num_blind / num_valid_pixels; 

% 【归一化】：盲区比例本身就是 0~1，无需额外处理
norm_blind = raw_blind_ratio;


% --- 指标 2: 均匀性 (Uniformity to Target 2.0) ---
% 原始值计算
Target_Value = 2.0;
mse_target = mean((roi_values - Target_Value).^2);
raw_rmse = sqrt(mse_target);

% 【归一化】：需要除以一个“理论最大误差”
% 假设最差情况是区域全是0，离2的距离是2；或者有极大热点。
% 设定 Max_RMSE = 2.5 作为保守的归一化分母
max_possible_rmse = 2.5; 
norm_uniformity = raw_rmse / max_possible_rmse;
% 防止溢出（虽然不太可能），截断到1
if norm_uniformity > 1, norm_uniformity = 1; end


% --- 指标 3: 避免混叠 (Aliasing) ---
% 原始值计算
diff_1 = 0;
limit_dist = 30; % 30mm = 3cm
% 收集所有有效距离
valid_paths = [];
for k = 1:num_genes_s
    dis = distance_matrix(k, :);
    valid_paths = [valid_paths, dis(dis > 0)];
end

% 计算惩罚
if ~isempty(valid_paths)
    for m = 1:length(valid_paths)-1
        for n = m+1:length(valid_paths)
            A = abs(valid_paths(m) - valid_paths(n));
            if A < limit_dist
                diff_1 = diff_1 + (limit_dist - A); 
            end
        end
    end
end
raw_aliasing = diff_1;

% 【归一化】：除以理论最大惩罚值
% 理论最大惩罚 = 路径组合数 * limit_dist (假设所有路径长度都一样，全混叠)
num_paths = length(valid_paths);
if num_paths > 1
    num_pairs = num_paths * (num_paths - 1) / 2;
else
    num_pairs = 1;
end
max_possible_aliasing = num_pairs * limit_dist; 
% 避免除以0
if max_possible_aliasing == 0, max_possible_aliasing = 1; end

norm_aliasing = raw_aliasing / max_possible_aliasing;


%% 5. 总目标函数输出 (加权求和)
% AHP 权重分配 (基于矩阵: 盲区>均匀>混叠)
% w1 (盲区) = 0.648
% w2 (均匀) = 0.230
% w3 (混叠) = 0.122

w1 = 0.648;
w2 = 0.230;
w3 = 0.122;

% 总目标函数 (取值范围 0~1，越小越好)
Y = (w1 * norm_blind) + (w2 * norm_uniformity) + (w3 * norm_aliasing);

end