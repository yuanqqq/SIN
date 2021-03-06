close all
clear
clc 

% plot eval curves 
% author: FEngting Yang 
% last modified: March 24th
% 
% Usage:
% 1. pick all the methods you interested in, and run them on the datasets
% 2. copy and run my_eval.sh in </path/to/superpixel-benchmark>/examples/bash
%    folder to get all the "results.csv" files for each method
% 3. set the res_path and load_path of each method correctly. We also provide
%    a toolkit named "copy_resCVS.py" to help organize the results. 
% 4. uncomment the methods slot in the following code, and run it
% 5. the result plots will be saved to the same dir of this .m file

%% ====  opt
formatSpec = ['%*s%f']; %ingnore the start string
sizeA = [1 Inf];

%===== data format of  *_save matrix  ==============
%  n_spixel  ASA   BR   BP    CO
% =====================================================
% 
% ===============
%  We provide code to plot all the method results we mentioned in the paper
%  uncomment them if needed 
% ================
% collect slic res
SLIC_res_path = './result_set/slic/nyu/' ;
num_list = [300:200:2300];
n_set = length(num_list);
Slic_save = zeros(n_set,5);
cnt = 1;

for i=1:n_set
    load_path = [SLIC_res_path num2str(num_list(i)) '/results.csv'];
    Slic_save(cnt,:) = loadcsv(load_path);
    cnt = cnt + 1;
end
% 
%% collect LSC res
% LSC_res_path = './result_set/lsc//' ;
% cnt = 1; 
% num_list = [50 80 100 150 200:100:1200]
% n_set = length(num_list)
% LSC_save = zeros(n_set,5);
% for i=1:n_set
%     load_path = [LSC_res_path num2str(num_list(i)) '/results.csv']; 
%     LSC_save(cnt,:) = loadcsv(load_path);
%     cnt = cnt + 1;
% end
% 
%% collect ERS res
n_set = 11
 ERS_res_path = './result_set/ers/nyu/' ;
 ERS_save = zeros(n_set,5);
 cnt = 1;
 for i=300:200:2300
     load_path = [ERS_res_path num2str(i) '/results.csv'];
     ERS_save(cnt,:) = loadcsv(load_path);
     cnt = cnt + 1;
 end
% 
%% collect SNIC res 
snic_res_path = './result_set/snic/nyu/' ;
SNIC_save = zeros(n_set,5);
cnt = 1; 
for i=300:200:2300
    load_path = [snic_res_path num2str(i) '/results.csv']; 
    SNIC_save(cnt,:) = loadcsv(load_path);
    cnt = cnt + 1;
end
% 
% %% collect ETPS
% etps_res_path = './result_set/fcn/nyu/' ;
% etps_save = zeros(n_set,5);
% cnt = 1; 
% num_list = [54 96 150 216 294 384 486 600 726 864 1014 1176 1350 1536 1944]; 
% for i=200:100:1100
%     load_path = [etps_res_path num2str(i) '/results.csv']; 
%     etps_save(cnt,:) = loadcsv(load_path);
%     cnt = cnt + 1;
% end
% 
%% collect SSN res
SSN_res_path = './result_set/ssn/nyu/' ;
num_list = [300:200:2300];
n_set = length(num_list);
SSN_save = zeros(n_set,5);
cnt = 1; 
for i=1:n_set
    load_path = [SSN_res_path  num2str(num_list(i)) '/results.csv']; 
    SSN_save(cnt,:) = loadcsv(load_path);
    cnt = cnt + 1;
end
% 
%% collect SEAL res
SEAL_res_path = './result_set/seal/nyu/' ;
SEAL_save = zeros(n_set,5);
cnt = 1; 
num_list = [300:200:2300];
n_set = length(num_list);
for i=1:n_set
    load_path = [SEAL_res_path  num2str(num_list(i)) '/results.csv']; 
    SEAL_save(cnt,:) = loadcsv(load_path); 
    cnt = cnt + 1;
end
%% collect fcn method res 
% please set the res_path appropriately before running 
fcn_res_path = './result_set/fcn/nyu/';
% num_list = [54 96 150 216 294 384 486 600 726 864 1014 1176 1350 1536 1944]; 
% num_list = [300 432];
num_list = [300 432 588 768 972 1200 1452 1728 2028 2352];
n_set = length(num_list);
fcn_save = zeros(n_set,5);
for i=1:n_set
    load_path = [fcn_res_path  num2str(num_list(i)) '/results.csv']; 
    fcn_save(i,:) = loadcsv(load_path);
end
% ===================================
%% collect our method res 
% please set the res_path appropriately before running 
our1l_res_path = './result_set/deconv/nyu/';
% num_list = [54 96 150 216 294 384 486 600 726 864 1014 1176 1350 1536 1944]; 
% num_list = [300 432];
num_list = [300 432 588 768 972 1200 1452 1728 2028 2352];
n_set = length(num_list);
Ours = zeros(n_set,5);
for i=1:n_set
    load_path = [our1l_res_path  num2str(num_list(i)) '/results.csv']; 
    Ours(i,:) = loadcsv(load_path);
end



%% plot ASA
% please uncomment the corresponding lines if you choose to plot that method 
h = zeros(1,1); 
figure;
h(1)=plot(Slic_save(:,1),Slic_save(:,2), 'b-*', 'MarkerSize',8); hold on;
h(2)=plot(SNIC_save(:,1),SNIC_save(:,2), 'g-p', 'MarkerFaceColor','g','MarkerSize',8); hold on;
% h(4)=plot(LSC_save(:,1),LSC_save(:,2), 'r-o', 'MarkerFaceColor','r','MarkerSize',8); hold on; 
h(3)=plot(ERS_save(:,1),ERS_save(:,2), 'r-+', 'MarkerSize',8); hold on; 
% h(6)=plot(etps_save(:,1),etps_save(:,2), 'g-x', 'MarkerSize',8); hold on; 
h(4)=plot(SEAL_save(:,1),SEAL_save(:,2), 'm-d','MarkerFaceColor','m', 'MarkerSize',8); hold on; 
h(5)=plot(SSN_save(:,1),SSN_save(:,2), 'c-s','MarkerFaceColor','c', 'MarkerSize',8); hold on; 
h(6)=plot(fcn_save(:,1),fcn_save(:,2), 'k-^', 'MarkerFaceColor', 'k', 'MarkerSize',8); hold on; 
h(7)=plot(Ours(:,1),Ours(:,2), 'r-o','MarkerFaceColor','r', 'MarkerSize',8); hold on; 


hold off;
lg = legend(h,'SLIC','SNIC','ERS', 'SEAL',  'SSN', 'FCN', 'Ours', ... 
          'Location','southeast');
% lg = legend(h,'Ours', 'Location','southeast');
set(lg,'FontSize',12, 'FontName', 'ArialMT');
xlim([300,2300])
ylim([0.89,0.96])
set(gca,'XTick',(300:400:2300),'FontName', 'ArialMT')
xlabel('Number of Superpixels')
ylabel('ASA Score')
set(gca,'FontSize',14,'FontName', 'ArialMT')


%% plot CO
% please uncomment the corresponding lines if you choose to plot that method 
h = zeros(1,1);
figure;
h(1)=plot(Slic_save(:,1),Slic_save(:,5), 'b-*','MarkerSize',8); hold on; 
h(2)=plot(SNIC_save(:,1),SNIC_save(:,5), 'g-p','MarkerFaceColor','g','MarkerSize',8); hold on;
% h(4)=plot(LSC_save(:,1),LSC_save(:,5), 'r-o','MarkerFaceColor','r','MarkerSize',8); hold on; 
h(3)=plot(ERS_save(:,1),ERS_save(:,5), 'r-+','MarkerSize',8); hold on; 
% h(6)=plot(etps_save(:,1),etps_save(:,5), 'g-x','MarkerSize',8); hold on; 
h(4)=plot(SEAL_save(:,1),SEAL_save(:,5), 'm-d','MarkerFaceColor','m','MarkerSize',8); hold on; 
h(5)=plot(SSN_save(:,1),SSN_save(:,5), 'c-s','MarkerFaceColor','c','MarkerSize',8); hold on; 
h(6)=plot(fcn_save(:,1),fcn_save(:,5), 'k-^','MarkerFaceColor','k','MarkerSize',8); hold on; 
h(7)=plot(Ours(:,1),Ours(:,5), 'r-o','MarkerFaceColor','r','MarkerSize',8); hold on; 
% h(7)=plot(fcn_save(:,1),fcn_save(:,5), 'g-x','MarkerSize',8); hold on;

hold off;
lg = legend(h,'SLIC','SNIC','ERS', 'SEAL',  'SSN', 'FCN', 'Ours',  ...
          'Location','southeast');
% lg = legend(h,'Ours', 'Location','southeast');

set(lg,'FontSize',12,'FontName', 'ArialMT');
xlim([300,2300])
ylim([0.1,0.55])
set(gca,'XTick',(300:400:2300),'FontName', 'ArialMT')
xlabel('Number of Superpixels')
ylabel('CO Score')
set(gca,'FontSize',14,'FontName', 'ArialMT')

%% plot BR-BP
% please uncomment the corresponding lines if you choose to plot that method 
h = zeros(1,1);
figure;
h(1)=plot(Slic_save(:,3),Slic_save(:,4), 'b-*','MarkerSize',8); hold on; 
h(2)=plot(SNIC_save(:,3),SNIC_save(:,4), 'g-p','MarkerFaceColor','g','MarkerSize',8); hold on;
% h(4)=plot(LSC_save(:,3),LSC_save(:,4), 'r-o','MarkerFaceColor','r','MarkerSize',8); hold on; 
h(3)=plot(ERS_save(:,3),ERS_save(:,4), 'r-+','MarkerSize',8); hold on;
% h(6)=plot(etps_save(:,3),etps_save(:,4), 'g-x','MarkerSize',8); hold on; 
h(4)=plot(SEAL_save(:,3),SEAL_save(:,4), 'm-d','MarkerFaceColor','m','MarkerSize',8); hold on; 
h(5)=plot(SSN_save(:,3),SSN_save(:,4), 'c-s','MarkerFaceColor','c','MarkerSize',8); hold on; 
h(6)=plot(fcn_save(:,3),fcn_save(:,4), 'k-^','MarkerFaceColor','k','MarkerSize',8); hold on; 
h(7)=plot(Ours(:,3),Ours(:,4), 'r-o','MarkerFaceColor','r','MarkerSize',8); hold on; 
% h(7)=plot(fcn_save(:,3),fcn_save(:,4), 'g-x','MarkerSize',8); hold on; 
hold off;
lg = legend(h, 'SLIC','SNIC','ERS', 'SEAL',  'SSN', 'FCN','Ours', ...
          'Location','northeast');
% lg = legend(h,'Ours', 'Location','northeast');

set(lg,'FontSize',12,'FontName', 'ArialMT');
xlim([0.8,0.95])
ylim([0.12,0.26])
set(gca,'XTick',(0.8:0.05:0.95),'FontName', 'ArialMT')
set(gca,'YTick',(0.12:0.02:0.26),'FontName', 'ArialMT')
xlabel('Boundary Recall')
ylabel('Boundary Precision')
set(gca,'FontSize',14,'FontName', 'ArialMT')
