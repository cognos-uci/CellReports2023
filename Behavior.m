clear;
addpath(genpath('G:\My Drive\Science\Analysis tools\Assisting Functions'));
addpath('G:\My Drive\Science\AtHomeContext\Data\Winter2021');
m_file_text=fileread(matlab.desktop.editor.getActiveFilename);
more_than_average_cuing=0;
DataFold='G:\My Drive\Science\TemporalContext\ContextRoom\log\';
load([pwd,'\Datasets\sublists']);
Recall=load([pwd,'\Datasets\RecallData']);
subnums=subs_atleast3itemrecall;
HoursPreSleep=[431.0000    7.0000;  432.0000    7.0000;  433.0000    7.0000;  434.0000    7.0000;  435.0000    4.0000;  436.0000    5.0000;  437.0000    6.5000;  438.0000    6.0000;  439.0000    8.7500;  440.0000    6.5000;  441.0000    6.0000;  442.0000    7.5000;  443.0000    7.5000;  444.0000    4.0000;  445.0000    8.0000;  446.0000    6.0000;  447.0000    5.5000;  448.0000    5.0000;  449.0000    5.0000;  450.0000    7.0000;  451.0000    4.5000;  452.0000    7.5000;  453.0000    5.5000;  454.0000    6.0000;  455.0000    6.0000;  456.0000    9.0000;  457.0000    4.0000;  458.0000    7.5000;  459.0000    3.5000;  460.0000    7.0000;  461.0000    5.5000;  462.0000    4.0000;  463.0000    4.0000; 464.0000    7.0000;  465.0000    5.0000;  466.0000    6.0000;  467.0000    3.5000;  468.0000    5.0000;  469.0000    6.5000;  470.0000    6.0000;  471.0000    7.5000;  472.0000    3.0000;  473.0000    5.0000;  474.0000    5.0000;  475.0000    6.5000;  476.0000    6.0000;  477.0000    6.0000;  478.0000    6.0000]; %The number of hours the subjects slept before coming in
HoursPreSleep=HoursPreSleep(ismember(HoursPreSleep(:,1),subnums),:);

RecalledItems=Recall.items(ismember(str2num(Recall.subnums),subnums),:);
filename=[sub_subset,'_Diary',num2str(size(ls([sub_subset,'_Diary*.txt']),1)+1),'.txt'];
diary (filename);diary on
for sub=1:length(subnums)
    fileID = fopen([DataFold,num2str(subnums(sub)),'\S',num2str(subnums(sub)),'-Training.txt'],'r');
    dataArray1 = textscan(fileID, '%f%f%f%f%f%f%f%f%f%f%*f%*s%[^\n\r]', 'Delimiter', '\t', 'TextType', 'string', 'HeaderLines' ,1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);    
    dataArray1(3:end+2)=dataArray1;
    dataArray1{1}=repmat(subnums(sub),size(dataArray1{1},1),1);
    dataArray1{2}=ones(size(dataArray1{1},1),1);

    fileID = fopen([DataFold,num2str(subnums(sub)),'\S',num2str(subnums(sub)),'-T2.txt'],'r');
    dataArray2 = textscan(fileID, '%f%f%f%f%f%f%f%f%f%f%*f%*s%[^\n\r]', 'Delimiter', '\t', 'TextType', 'string', 'HeaderLines' ,1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    fclose(fileID);    
    dataArray2(3:end+2)=dataArray2;
    dataArray2{1}=repmat(subnums(sub),size(dataArray2{1},1),1);
    dataArray2{2}=repmat(2,size(dataArray2{1},1),1);
    
    dataArray1(18)=dataArray1(13);    dataArray1{13}=nan(size(dataArray1{1},1),1);    dataArray1{14}=zeros(size(dataArray1{1},1),1);    dataArray1{15}=zeros(size(dataArray1{1},1),1);  dataArray1{16}=zeros(size(dataArray1{1},1),1);  dataArray1{17}=zeros(size(dataArray1{1},1),1);    
    dataArray2(18)=dataArray2(13);    dataArray2{13}=nan(size(dataArray2{1},1),1);    dataArray2{14}=zeros(size(dataArray2{1},1),1);    dataArray2{15}=zeros(size(dataArray2{1},1),1);  dataArray2{16}=zeros(size(dataArray2{1},1),1);  dataArray2{17}=zeros(size(dataArray2{1},1),1);    
    Scenes=setdiff(unique(dataArray1{5}),0);
    for ii=Scenes'
        dataArray1{13}(dataArray1{5}==ii)=dataArray1{4}(find(dataArray1{5}==ii,1));
        dataArray2{13}(dataArray2{5}==ii)=dataArray1{4}(find(dataArray1{5}==ii,1));    
        if ii<=18
            dataArray1{17}(dataArray1{5}==ii)=double(ismember(dataArray1{6}(dataArray1{5}==ii),RecalledItems{sub,ii}));
            dataArray2{17}(dataArray2{5}==ii)=double(ismember(dataArray2{6}(dataArray2{5}==ii),RecalledItems{sub,ii}));
        else
            dataArray1{17}(dataArray1{5}==ii)=NaN;
            dataArray2{17}(dataArray2{5}==ii)=NaN;
        end
    end
    fileID = fopen([DataFold,num2str(subnums(sub)),'\TMR_list_for_',num2str(subnums(sub)),'.txt'],'r');
    TMRed = textscan(fileID, '%f%[^\n\r]', 'Delimiter', {''}, 'TextType', 'string',  'ReturnOnError', false);
    fclose(fileID);
    dataArray1{14}(ismember(dataArray1{6}+(dataArray1{5}-1)*4,TMRed{1}))=1;    dataArray2{14}(ismember(dataArray2{6}+(dataArray2{5}-1)*4,TMRed{1}))=1;    
    dataArray1{15}(ismember(dataArray1{5},unique(dataArray1{5}(dataArray1{14}==1))))=1;    dataArray2{15}(ismember(dataArray2{5},unique(dataArray2{5}(dataArray2{14}==1))))=1;
    dataArray1{16}(ismember(dataArray1{13},unique(dataArray1{13}(dataArray1{14}==1))))=1;    dataArray2{16}(ismember(dataArray2{13},unique(dataArray2{13}(dataArray2{14}==1))))=1;
    
    tmpData = [table(dataArray1{[1:11,13:end-1]}, 'VariableNames', {'sub','T1_2','Stage','Block','Set','Picture','RepNum','TrueLocX','TrueLocY','UserLocX','UserLocY','LearningBlock','CuedItem','CuedScene','CuedBlock','RecalledOrNot'});...
               table(dataArray2{[1:11,13:end-1]}, 'VariableNames', {'sub','T1_2','Stage','Block','Set','Picture','RepNum','TrueLocX','TrueLocY','UserLocX','UserLocY','LearningBlock','CuedItem','CuedScene','CuedBlock','RecalledOrNot'})];
    tmpData.CuedItem(tmpData.Set==0 | tmpData.Set==19)=NaN;
    tmpData.CuedScene(tmpData.Set==0 | tmpData.Set==19)=NaN;
    tmpData.CuedBlock(tmpData.Set==0 | tmpData.Set==19)=NaN;
    clearvars fileID dataArray1 dataArray2;
    if sub==1
        Data=tmpData;
    else
        Data=[Data;tmpData];
    end
end
clearvars tmpData;
Data.Stage(Data.Stage==2)=0;
Data.Stage(Data.Stage==4)=1;
Data.Stage(Data.Stage==5)=2;
Data.UniqueID(:,1)=Data.sub*1000+Data.Set*10+Data.Picture;
Data.UniqueSetID(:,1)=Data.sub*1000+Data.Set*10;
Data.UniqueBlockID(:,1)=Data.sub*1000+Data.LearningBlock*10;

NaNidx=find(Data.UserLocX==-1 & Data.UserLocY==-1);Data.UserLocY(NaNidx)=nan;Data.UserLocX(NaNidx)=nan;
Data.Diff=((Data.UserLocX-Data.TrueLocX).^2+(Data.UserLocY-Data.TrueLocY).^2).^0.5;
OrigData=Data;
Data=Data(~isnan(Data.Diff) & Data.Set~=19 & Data.Picture<=72,:);
Data.T2Diff(:,1)=nan;
Data.T2T1Benefit(:,1)=nan;
Data.T2T1BenefitPerc(:,1)=nan;

Data.Correct=double(Data.Diff<=100);
Data.T2Correct(:,1)=nan;
Data.T2T1CorrectBenefit(:,1)=nan;

Data.IsSwap(:,1)=0;
Data.T2IsSwap(:,1)=nan;
Data.T2T1BenefitSwap(:,1)=nan;

Data.DiffIfClosest=Data.Diff;
Data.T2DiffIfClosest(:,1)=nan;
Data.T2T1BenefitIfClosest(:,1)=nan;
Data.T2T1BenefitIfClosestPerc(:,1)=nan;

Data.DiffToClosest=Data.Diff;
Data.T2DiffToClosest(:,1)=nan;
Data.T2T1BenefitToClosest(:,1)=nan;

Data.DiffFromSceneMass(:,1)=nan;
Data.T2DiffFromSceneMass(:,1)=nan;
Data.T2T1BenefitDiffFromSceneMass(:,1)=nan;

Data.DiffFromBlockMass(:,1)=nan;
Data.T2DiffFromBlockMass(:,1)=nan;
Data.T2T1BenefitDiffFromBlockMass(:,1)=nan;

Data.Azimuth=atan2((Data.UserLocX-Data.TrueLocX),(Data.UserLocY-Data.TrueLocY));
Data.T2Azimuth(:,1)=nan;
Data.T2T1BenefitAzimuth(:,1)=nan;

Data.AzimuthIfClosest(:,1)=Data.Azimuth;
Data.T2AzimuthIfClosest(:,1)=nan;
Data.T2T1BenefitAzimuthIfClosest(:,1)=nan;

Data.AzimuthFromSceneMass(:,1)=nan;
Data.T2AzimuthFromSceneMass(:,1)=nan;
Data.T2T1BenefitAzimuthFromSceneMass(:,1)=nan;

Data.AzimuthFromBlockMass(:,1)=nan;
Data.T2AzimuthFromBlockMass(:,1)=nan;
Data.T2T1BenefitAzimuthFromBlockMass(:,1)=nan;

Data.T2T1UserDist(:,1)=nan;

for sub=1:size(subnums,1)
    for scene=1:18
        scenemass=[mean(Data.TrueLocX(find(Data.sub==subnums(sub) & Data.Set==scene,4))),mean(Data.TrueLocY(find(Data.sub==subnums(sub) & Data.Set==scene,4)))];
        Data.DiffFromSceneMass(Data.sub==subnums(sub) & Data.Set==scene)=((Data.UserLocX(Data.sub==subnums(sub) & Data.Set==scene)-scenemass(1)).^2+(Data.UserLocY(Data.sub==subnums(sub) & Data.Set==scene)-scenemass(2)).^2).^0.5;
        Data.AzimuthFromSceneMass(Data.sub==subnums(sub) & Data.Set==scene)=atan2((Data.UserLocX(Data.sub==subnums(sub) & Data.Set==scene)-scenemass(1)),(Data.UserLocY(Data.sub==subnums(sub) & Data.Set==scene)-scenemass(2)));
    end
    for block=1:9
        blockmass=[mean(Data.TrueLocX(find(Data.sub==subnums(sub) & Data.LearningBlock==block,8))),mean(Data.TrueLocY(find(Data.sub==subnums(sub) & Data.LearningBlock==block,8)))];
        Data.DiffFromBlockMass(Data.sub==subnums(sub) & Data.LearningBlock==block)=((Data.UserLocX(Data.sub==subnums(sub) & Data.LearningBlock==block)-blockmass(1)).^2+(Data.UserLocY(Data.sub==subnums(sub) & Data.LearningBlock==block)-blockmass(2)).^2).^0.5;
        Data.AzimuthFromBlockMass(Data.sub==subnums(sub) & Data.LearningBlock==block)=atan2((Data.UserLocX(Data.sub==subnums(sub) & Data.LearningBlock==block)-blockmass(1)),(Data.UserLocY(Data.sub==subnums(sub) & Data.LearningBlock==block)-blockmass(2)));
    end
end
for ii=1:size(Data,1)
    idxs=find(Data.sub==Data.sub(ii) & Data.Set==Data.Set(ii) & Data.Picture~=Data.Picture(ii) & Data.Stage==1,3);
    diffs=sum((table2array(Data(idxs,8:9))-repmat(table2array(Data(ii,10:11)),3,1)).^2,2).^0.5;
    if ~isempty(find(diffs<Data.DiffIfClosest(ii), 1))
        Data.DiffIfClosest(ii)=nan;
        Data.AzimuthIfClosest(ii)=nan;
        Data.IsSwap(ii)=1;
        Data.DiffToClosest(ii)=min(diffs);
    end    
end
for ii=find(Data.Stage>0)'
    if Data.Stage(ii)==1
        idx=Data.UniqueID==Data.UniqueID(ii) & Data.Stage==2;
        Data.T2Diff(ii)=Data.Diff(idx);
        Data.T2IsSwap(ii)=Data.IsSwap(idx);
        Data.T2Azimuth(ii)=Data.Azimuth(idx);
        Data.T2Correct(ii)=Data.Correct(idx);
        Data.T2DiffFromSceneMass(ii)=Data.DiffFromSceneMass(idx);
        Data.T2DiffFromBlockMass(ii)=Data.DiffFromBlockMass(idx);
        Data.T2AzimuthFromSceneMass(ii)=Data.AzimuthFromSceneMass(idx);
        Data.T2AzimuthFromBlockMass(ii)=Data.AzimuthFromBlockMass(idx);
        Data.T2DiffIfClosest(ii)=Data.DiffIfClosest(idx);
        Data.T2AzimuthIfClosest(ii)=Data.AzimuthIfClosest(idx);
        Data.T2DiffToClosest(ii)=Data.DiffToClosest(idx);
        Data.T2T1Benefit(ii)=Data.Diff(ii)-Data.Diff(idx);
        Data.T2T1BenefitPerc(ii)=100*(Data.Diff(ii)-Data.Diff(idx))./Data.Diff(ii);
        Data.T2T1BenefitIfClosest(ii)=Data.DiffIfClosest(ii)-Data.DiffIfClosest(idx);
        Data.T2T1BenefitIfClosestPerc(ii)=100*(Data.DiffIfClosest(ii)-Data.DiffIfClosest(idx))./Data.DiffIfClosest(ii);
        Data.T2T1BenefitDiffFromSceneMass(ii)=Data.DiffFromSceneMass(ii)-Data.DiffFromSceneMass(idx);
        Data.T2T1BenefitDiffFromBlockMass(ii)=Data.DiffFromBlockMass(ii)-Data.DiffFromBlockMass(idx);
        Data.T2T1BenefitToClosest(ii)=Data.DiffToClosest(ii)-Data.DiffToClosest(idx);
        Data.T2T1UserDist(ii)=((Data.UserLocX(ii)-Data.UserLocX(idx)).^2+(Data.UserLocY(ii)-Data.UserLocY(idx)).^2).^0.5;
        Data.T2T1CorrectBenefit(ii)=Data.Correct(idx)-Data.Correct(ii);
        Data.T2T1BenefitSwap(ii)=Data.IsSwap(ii)-Data.IsSwap(idx);
    end
end
try
    Data.T2T1BenefitAzimuth(:,1)=circ_dist(Data.Azimuth,Data.T2Azimuth);
    Data.T2T1BenefitAzimuthIfClosest(:,1)=circ_dist(Data.AzimuthIfClosest,Data.T2AzimuthIfClosest);
    Data.T2T1BenefitAzimuthFromSceneMass(:,1)=circ_dist(Data.AzimuthFromSceneMass,Data.T2AzimuthFromSceneMass);
    Data.T2T1BenefitAzimuthFromBlockMass(:,1)=circ_dist(Data.AzimuthFromBlockMass,Data.T2AzimuthFromBlockMass);
end

TestData=Data(Data.Stage==1,:);
TestData.Cond(:,1)=TestData.CuedItem(:,1)+TestData.CuedScene(:,1)+TestData.CuedBlock(:,1)+1;

for sub=1:size(subnums,1)
    idx=find(TestData.sub==subnums(sub));
    [~,~,TestData.T2T1BenefitIfClosest_Resid(idx,1)]=regress(TestData.T2T1BenefitIfClosest(idx),[ones(length(idx),1),TestData.DiffIfClosest(idx)]);
end

TestData.Set=categorical(TestData.Set);
TestData.Picture=categorical(TestData.Picture);
TestData.sub=categorical(TestData.sub);
TestData.CuedItem=logical(TestData.CuedItem);
TestData.CuedScene=logical(TestData.CuedScene);
TestData.CuedBlock=logical(TestData.CuedBlock);


%% %%%%%%%%%%%%%%%%%%%%% Analyses %%%%%%%%%%%%%%%%%%%%%%%%
        %% %%%%%%%%%%%%%%%%%%%%% Descriptive Stats %%%%%%%%%%%%%%%%%%%%%%%%
            %%
                disp('Check the percent of trials in training failing because of wrong YN answer (per sub)');
                clearvars percent_wrong_YN;
                for sub=1:length(subnums)
                    percent_wrong_YN(sub)=100*length(find(isnan(OrigData.UserLocX(OrigData.sub==subnums(sub) & OrigData.Stage==0 & OrigData.RepNum>0 & OrigData.Picture<=72))))/length(OrigData.UserLocX(OrigData.sub==subnums(sub) & OrigData.Stage==0 & OrigData.RepNum>0 & OrigData.Picture<=72));
                end
                disp([num2str(mean(percent_wrong_YN)),'+-',num2str(std(percent_wrong_YN)/sqrt(length(subnums))),'(SE)']);
            %%
                disp('Check the number of reps needed for an item to drop out (per sub)');
                clearvars num_reps_required;
                for sub=1:length(subnums)
                    tmpvec=[];
                    for ii=unique(Data.UniqueID(Data.sub==subnums(sub)))'
                        tmpvec=[tmpvec;max(Data.RepNum(Data.sub==subnums(sub) & Data.Stage==0 & Data.UniqueID==ii))];
                    end
                    num_reps_required(sub)=nanmean(tmpvec);
                end
                disp([num2str(mean(num_reps_required)),'+-',num2str(std(num_reps_required)/sqrt(length(subnums))),'(SE)']);
            %%
                disp('Check the pre-sleep error in pixels');
                for sub=1:length(subnums)
                    pre_sleep_error(sub)=nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1));
                end
                disp([num2str(mean(pre_sleep_error)),'+-',num2str(std(pre_sleep_error)/sqrt(length(subnums))),'(SE)']);
               
            %%
                disp('Check the pre-sleep error in pixels INCLUDING SWAPPED ITEMS');
                for sub=1:length(subnums)
                    pre_sleep_error(sub)=nanmean(Data.Diff(Data.sub==subnums(sub) & Data.Stage==1));
                end
                disp([num2str(mean(pre_sleep_error)),'+-',num2str(std(pre_sleep_error)/sqrt(length(subnums))),'(SE)']);
            %%
                disp('Check the pre-sleep & post-sleep swap errors');
                for sub=1:length(subnums)
                    pre_sleep_swaps(sub,:)=[nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==1)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedBlock==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedBlock==1 & Data.CuedScene==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedItem==1))];
                    post_sleep_swaps(sub,:)=[nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==2)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedBlock==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedBlock==1 & Data.CuedScene==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.IsSwap(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedItem==1))];
                end
                disp(['pre swap rate is ',num2str(mean(pre_sleep_swaps(:,1))),'+-',num2str(std(pre_sleep_swaps(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['post swap rate is ',num2str(mean(post_sleep_swaps(:,1))),'+-',num2str(std(post_sleep_swaps(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['pre swap rate per condition (0,1,2,3,4):']);disp(mean(pre_sleep_swaps(:,2:end),1));disp(std(pre_sleep_swaps(:,2:end),[],1)/sqrt(length(subnums)));
                disp(['post swap rate per condition (0,1,2,3,4):']);disp(mean(post_sleep_swaps(:,2:end),1));disp(std(post_sleep_swaps(:,2:end),[],1)/sqrt(length(subnums)));
            %%
                disp('Check the pre-sleep & post-sleep error in pixels (three conditions)');
                clearvars pre_sleep_errors_detailed post_sleep_errors_detailed;
                for sub=1:length(subnums)
                    pre_sleep_errors_detailed(sub,:)=[nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedScene==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedItem==1))];
                    post_sleep_errors_detailed(sub,:)=[nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedScene==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedItem==1))];
                end
                disp(['pre error in pixels is ',num2str(mean(pre_sleep_errors_detailed(:,1))),'+-',num2str(std(pre_sleep_errors_detailed(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['post error in pixels is ',num2str(mean(post_sleep_errors_detailed(:,1))),'+-',num2str(std(post_sleep_errors_detailed(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['pre error in pixels per condition (1,2,3):']);disp(mean(pre_sleep_errors_detailed(:,2:end),1));disp(std(pre_sleep_errors_detailed(:,2:end),[],1)/sqrt(length(subnums)));
                disp(['post error in pixels per condition (1,2,3):']);disp(mean(post_sleep_errors_detailed(:,2:end),1));disp(std(post_sleep_errors_detailed(:,2:end),[],1)/sqrt(length(subnums)));
                pre_sleep_errors_table=array2table(pre_sleep_errors_detailed(:,2:end))
                factornames={'Condition'};
                within=table({'1';'2';'3'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                rm = fitrm(pre_sleep_errors_table,'Var1-Var3~1','WithinDesign',within);
                disp('Repeated Measures ANOVA across participants to test the effects of cnodition to pre-sleep error:')
                ranova(rm, 'WithinModel','Condition')
                
                TestData2=TestData;TestData2.Cond=categorical(TestData2.Cond);
                TestData2.Cond=mergecats(TestData2.Cond,["1","2"]);
                TestData2.Cond=renamecats(TestData2.Cond,"3","2");
                TestData2.Cond=renamecats(TestData2.Cond,"4","3");
                lm=fitlme(TestData2,'DiffIfClosest ~ Cond + (Cond|sub)');
                disp('Same in LM:')
                anova(lm)
            %% Looking at pre-post difference for different conditions (classic analysis) in an RM - only cued vs non-cued in non-cued context
            
                errors_table=array2table([pre_sleep_errors_detailed(:,[2,4]),post_sleep_errors_detailed(:,[2,4])]);
                factornames={'Condition','PrePost'};
                within=table({'1';'3';'1';'3'},{'1';'1';'2';'2'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                within.PrePost = categorical(within.PrePost);
                rm2 = fitrm(errors_table,'Var1-Var4~1','WithinDesign',within);
                [ranovatbl] = ranova(rm2, 'WithinModel','Condition*PrePost')
            
            %% Looking at pre-post difference for different conditions (classic analysis) in an RM - only non-cued in vs out of context
            
                errors_table=array2table([pre_sleep_errors_detailed(:,[3,4]),post_sleep_errors_detailed(:,[3,4])]);
                factornames={'Condition','PrePost'};
                within=table({'2';'3';'2';'3'},{'1';'1';'2';'2'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                within.PrePost = categorical(within.PrePost);
                rm2 = fitrm(errors_table,'Var1-Var4~1','WithinDesign',within);
                [ranovatbl] = ranova(rm2, 'WithinModel','Condition*PrePost')

            %% For completeness - same for cued vs non-cued in a cued set
            
                errors_table=array2table([pre_sleep_errors_detailed(:,[2,3]),post_sleep_errors_detailed(:,[2,3])]);
                factornames={'Condition','PrePost'};
                within=table({'1';'2';'1';'2'},{'1';'1';'2';'2'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                within.PrePost = categorical(within.PrePost);
                rm2 = fitrm(errors_table,'Var1-Var4~1','WithinDesign',within);
                [ranovatbl] = ranova(rm2, 'WithinModel','Condition*PrePost')
                
            %% Looking at pre-post difference for different conditions (classic analysis) in both an RM and a LM
            
                errors_table=array2table([pre_sleep_errors_detailed(:,2:end),post_sleep_errors_detailed(:,2:end)]);
                factornames={'Condition','PrePost'};
                within=table({'1';'2';'3';'1';'2';'3'},{'1';'1';'1';'2';'2';'2'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                within.PrePost = categorical(within.PrePost);
                rm2 = fitrm(errors_table,'Var1-Var6~1','WithinDesign',within);
                [ranovatbl] = ranova(rm2, 'WithinModel','Condition*PrePost')
            
                TestData2=TestData;TestData2.Cond=categorical(TestData2.Cond);
                lm=fitlme(TestData2,'T2T1Benefit ~ Cond + (Cond|sub)');
                anova(lm)                
            %%
                disp('Check the pre-sleep & post-sleep error in pixels (four conditions)');
                clearvars pre_sleep_errors_detailed post_sleep_errors_detailed;
                for sub=1:length(subnums)
                    pre_sleep_errors_detailed(sub,:)=[nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedBlock==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedBlock==1 & Data.CuedScene==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==1 & Data.CuedItem==1))];
                    post_sleep_errors_detailed(sub,:)=[nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedBlock==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedBlock==1 & Data.CuedScene==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedScene==1 & Data.CuedItem==0)),nanmean(Data.DiffIfClosest(Data.sub==subnums(sub) & Data.Stage==2 & Data.CuedItem==1))];
                end
                disp(['pre error in pixels is ',num2str(mean(pre_sleep_errors_detailed(:,1))),'+-',num2str(std(pre_sleep_errors_detailed(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['post error in pixels is ',num2str(mean(post_sleep_errors_detailed(:,1))),'+-',num2str(std(post_sleep_errors_detailed(:,1))/sqrt(length(subnums))),'(SE)']);
                disp(['pre error in pixels per condition (1,2,3,4):']);disp(mean(pre_sleep_errors_detailed(:,2:end),1));disp(std(pre_sleep_errors_detailed(:,2:end),[],1)/sqrt(length(subnums)));
                disp(['post error in pixels per condition (1,2,3,4):']);disp(mean(post_sleep_errors_detailed(:,2:end),1));disp(std(post_sleep_errors_detailed(:,2:end),[],1)/sqrt(length(subnums)));
                pre_sleep_errors_table=array2table(pre_sleep_errors_detailed(:,2:end))
                factornames={'Condition'};
                within=table({'1';'2';'3';'4'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                rm = fitrm(pre_sleep_errors_table,'Var1-Var4~1','WithinDesign',within);
                disp('Repeated Measures ANOVA across participants to test the effects of cnodition to pre-sleep error:')
                ranova(rm, 'WithinModel','Condition')
                
                TestData2=TestData;TestData2.Cond=categorical(TestData2.Cond);
                lm=fitlme(TestData2,'DiffIfClosest ~ Cond + (Cond|sub)');
                disp('Same in LM:')
                anova(lm)
                
            %% Looking at pre-post difference for different conditions - 4 condition version (classic analysis) in both an RM and a LM
            
                errors_table=array2table([pre_sleep_errors_detailed(:,2:end),post_sleep_errors_detailed(:,2:end)]);
                factornames={'Condition','PrePost'};
                within=table({'1';'2';'3';'4';'1';'2';'3';'4'},{'1';'1';'1';'1';'2';'2';'2';'2'},'VariableNames',factornames);
                within.Condition = categorical(within.Condition);
                within.PrePost = categorical(within.PrePost);
                rm2 = fitrm(errors_table,'Var1-Var8~1','WithinDesign',within);
                [ranovatbl] = ranova(rm2, 'WithinModel','Condition*PrePost')
            
                TestData2=TestData;TestData2.Cond=categorical(TestData2.Cond);
                lm=fitlme(TestData2,'T2T1Benefit ~ Cond + (Cond|sub)');
                anova(lm)
                
            %%
                disp('Check the recall Data - general stats');
                disp([num2str(mean(mean(cellfun(@(x)length(x),RecalledItems),2))),'+-',num2str(std(mean(cellfun(@(x)length(x),RecalledItems),2))/sqrt(length(subnums))),'(SE)']);

                disp('Check whether cued context are remembered better');               
                numitemsrecalled=cellfun(@(x)length(x),RecalledItems);
                for sub=1:length(subnums)
                    recall_for_cued(sub,1)=mean(numitemsrecalled(sub,unique(OrigData.Set(OrigData.sub==subnums(sub) & OrigData.CuedScene==1))));
                    recall_for_cued(sub,2)=mean(numitemsrecalled(sub,setdiff(1:18,unique(OrigData.Set(OrigData.sub==subnums(sub) & OrigData.CuedScene==1)))));
                end
                disp(['Mean for cued/uncued is: ',num2str(mean(recall_for_cued(:,1))),', ',num2str(mean(recall_for_cued(:,2)))]);
                [~,p,~,stats]=ttest(recall_for_cued(:,1)-recall_for_cued(:,2));
                disp(['two way p is t(',num2str(stats.df),')=',num2str(stats.tstat),', p=',num2str(p)]);
                
                disp('Check whether cued items in cued contexts are remembered earlier');
                average_ord_across_subs=zeros(0,4);
                for sub=1:length(subnums)
                    cuedscenes=unique(OrigData.Set(OrigData.sub==subnums(sub) & OrigData.CuedScene==1));
                    average_ord=nan(6,4);
                    for scene=1:length(cuedscenes)
                        ord=RecalledItems{sub,cuedscenes(scene)};
                        items=[OrigData.Picture(find(OrigData.sub==subnums(sub) & isnan(OrigData.UserLocX) & OrigData.Set==cuedscenes(scene),4)),OrigData.CuedItem(find(OrigData.sub==subnums(sub) & isnan(OrigData.UserLocX) & OrigData.Set==cuedscenes(scene),4))];
                        for ii=1:length(ord)
                            ord(ii)=items(items(:,1)==ord(ii),2);
                        end
                        average_ord(scene,1:length(ord))=ord;
                        
                    end
                    percent_cued_out_of_recalled(sub)=nanmean(nansum(average_ord,2)./sum(~isnan(average_ord),2));
                    average_ord_across_subs(sub,:)=nanmean(average_ord,1);
                end
                disp('The chance of being cued per order place is:');
                disp(nanmean(average_ord_across_subs,1));
                disp('The probability of the object that is cued to be recalled')
                disp(mean(percent_cued_out_of_recalled));
                [~,p]=ttest2(percent_cued_out_of_recalled,0.5);
                disp(['Difference from 0.5 is p=',num2str(p)]);
    
            %%
                disp('Looking at SSS and boxtime') % 1 IS FULLY AWAKE
                SSS=nan(length(subnums),2);
                for sub=1:length(subnums)
                    counter=1;ind=1;
                    origfilename=[DataFold,num2str(subnums(sub)),'\',num2str(subnums(sub)),'-ContextRoom.log'];
                    filename=origfilename;
                    while counter<=2
                        if exist(filename,'file')
                            SSSfile=fileread(filename);
                            SSS(sub,counter)=str2double(SSSfile(strfind(SSSfile, 'SSS')+4));
                            
                            if ~isnan(SSS(sub,counter)) && SSS(sub,counter)~=0 
                                fileID = fopen(filename,'r');
                                textscan(fileID, '%[^\n\r]', 3, 'WhiteSpace', '', 'ReturnOnError', false, 'EndOfLine', '\r\n');
                                dataArray = textscan(fileID, '%*s%*s%*s%s%f%*s%*s%*s%*s%*s%*s%*s%*s%[^\n\r]', 'Delimiter', '\t', 'EmptyValue' ,NaN,'ReturnOnError', false);
                                fclose(fileID);
                                RTDATA = table(dataArray{1:end-1}, 'VariableNames', {'Code','Time'});
                                RTDATA.Code(end)={'3'};
                                Press1=find(cellfun(@(x)strcmp(x,'Wake-up: Press 1'),RTDATA.Code));
                                Press3=find(cellfun(@(x)strcmp(x,'Wake-up: Press 3'),RTDATA.Code));
                                Pressed3=find(cellfun(@(x)strcmp(x,'128'),RTDATA.Code));
                                Pressed1=find(cellfun(@(x)strcmp(x,'32'),RTDATA.Code));
                                RedBoxData{sub,counter}=[[Press1,ones(length(Press1),1)];[Press3,3*ones(length(Press3),1)]];
                                [~,a]=sort(RedBoxData{sub,counter}(:,1));
                                RedBoxData{sub,counter}=RedBoxData{sub,counter}(a,:);
                                for ii=1:size(RedBoxData{sub,counter},1)
                                    if Pressed1(find(Pressed1>RedBoxData{sub,counter}(ii,1),1))<Pressed3(find(Pressed3>RedBoxData{sub,counter}(ii,1),1))
                                        if ii==size(RedBoxData{sub,counter},1) || Pressed1(find(Pressed1>RedBoxData{sub,counter}(ii,1),1))<RedBoxData{sub,counter}(ii+1,1)
                                            RedBoxData{sub,counter}(ii,3:4)=[1,Pressed1(find(Pressed1>RedBoxData{sub,counter}(ii,1),1))];
                                        else
                                            RedBoxData{sub,counter}(ii,3:4)=nan;
                                        end
                                    else
                                        if ii==size(RedBoxData{sub,counter},1) || Pressed3(find(Pressed3>RedBoxData{sub,counter}(ii,1),1))<RedBoxData{sub,counter}(ii+1,1)
                                            RedBoxData{sub,counter}(ii,3:4)=[3,Pressed3(find(Pressed3>RedBoxData{sub,counter}(ii,1),1))];
                                        else
                                            RedBoxData{sub,counter}(ii,3:4)=nan;
                                        end
                                    end
                                end
                                for ii=1:size(RedBoxData{sub,counter},1)
                                    if ~isnan(RedBoxData{sub,counter}(ii,4)) && RedBoxData{sub,counter}(ii,2)==RedBoxData{sub,counter}(ii,3)
                                        RedBoxData{sub,counter}(ii,5)=(RTDATA.Time(RedBoxData{sub,counter}(ii,4))-RTDATA.Time(RedBoxData{sub,counter}(ii,1)))/10;
                                    else
                                        RedBoxData{sub,counter}(ii,5)=nan;
                                    end
                                end                              
                                
                                counter=counter+1;
                            end
                        end
                        filename=[origfilename(1:end-4),num2str(ind),origfilename(end-3:end)];
                        ind=ind+1;
                    end
                end
                disp('SSS averages (and then SE) before and after sleep:')
                disp(mean(SSS,1))
                disp(nanstd(SSS,1)./sqrt(sum(~isnan(SSS),1)))
                [~,p,~,stats]=ttest(SSS(:,1),SSS(:,2));
                disp(['two way p is t(',num2str(stats.df),')=',num2str(stats.tstat),', p=',num2str(p)]);
 
                disp('Looking at BoxTime')
                for ii=1:size(RedBoxData,1),attemptsandtime(ii,:)=[size(RedBoxData{ii,1},1),size(RedBoxData{ii,2},1),mean(RedBoxData{ii,1}(find(~isnan(RedBoxData{ii,1}(:,5)),8,'last'),5)),mean(RedBoxData{ii,2}(find(~isnan(RedBoxData{ii,2}(:,5)),8,'last'),5))];end
                disp('SSS averages (and then SE) before and after sleep: (num attempts per, num attempts post, RT per, RT post)')
                disp(nanmean(attemptsandtime,1))
                disp(nanstd(attemptsandtime,1)./sqrt(sum(~isnan(attemptsandtime),1)))
                [~,p,~,stats]=ttest(attemptsandtime(:,2)-attemptsandtime(:,1));
                disp(['Attempts: two way p is t(',num2str(stats.df),')=',num2str(stats.tstat),', p=',num2str(p)]);
                [~,p,~,stats]=ttest(attemptsandtime(:,4)-attemptsandtime(:,3));
                disp(['RT: two way p is t(',num2str(stats.df),')=',num2str(stats.tstat),', p=',num2str(p)]);

                %% correlate time asleep the night before with behavior pre-nap:
                [r,p]=corr(num_reps_required',HoursPreSleep(:,2));
                disp(['The correlation between the hours asleep and the number of repetitions required for learning is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(pre_sleep_error',HoursPreSleep(:,2));
                disp(['The correlation between the hours asleep and the average pre-sleep error is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(num_reps_required',attemptsandtime(:,1));
                disp(['The correlation between the number of attempts of the redbox and the number of repetitions required for learning is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(pre_sleep_error',attemptsandtime(:,1));
                disp(['The correlation between the number of attempts of the redbox and the average pre-sleep error is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(num_reps_required',attemptsandtime(:,3));
                disp(['The correlation between the RT of the redbox and the number of repetitions required for learning is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(pre_sleep_error',attemptsandtime(:,3));
                disp(['The correlation between the RT of the redbox and the average pre-sleep error is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(num_reps_required',SSS(:,1));
                disp(['The correlation between the SSS and the number of repetitions required for learning is ',num2str(r),'(p = ',num2str(p),')']);
                [r,p]=corr(pre_sleep_error',SSS(:,1));
                disp(['The correlation between the SSS and the average pre-sleep error is ',num2str(r),'(p = ',num2str(p),')']);

                %% Median error rate across ppt and conditions; chance level within a circle
                median([TestData.Diff;TestData.T2Diff])

                t1=randi((540-50)*2,1e7,2)-(540-50);
                t1=t1((sum(t1.^2,2).^0.5)>50,:);if mod(size(t1,1),2)==1,t1=t1(1:end-1,:);end
                t2=mean(sum((t1(1:end/2,:)-t1(end/2+1:end,:)).^2,2).^0.5);

                %% Sounds presentation per participant
                load([pwd,'\Datasets\SoundPresPerStage.mat']);% different subselection of subjects, set by the script 'soundsineachstage.m'
                for sub=1:length(subnums)
                    Sound76pres(sub)=datatab.N2(datatab.subnum==subnums(sub) & datatab.sndID==76)+datatab.N3(datatab.subnum==subnums(sub) & datatab.sndID==76);
                    total_num_sounds_per_cond(sub,:)=[sum(datatab.W(datatab.subnum==subnums(sub))),sum(datatab.N1(datatab.subnum==subnums(sub))),sum(datatab.N2(datatab.subnum==subnums(sub))),sum(datatab.N3(datatab.subnum==subnums(sub))),sum(datatab.R(datatab.subnum==subnums(sub)))];
                end
                disp(['average number of 76 sounds was ',num2str(mean(Sound76pres))]);
                
                %%
                try
                    T2difftxt='T2DiffIfClosest';
                    T1difftxt='DiffIfClosest';
                    T2T1benefittxt='T2T1BenefitIfClosest';
                    add_text_Cond=' NonRecall included:';                
                    if ~isempty(T2difftxt)
                        disp(['Splitting the Data by T1 results (default: median split) and considering the two halves separately:',add_text_Cond])
                        fractions=2;
                        for ii=1:fractions
                            TestData2=TestData;TestData2.Cond=ordinal(TestData2.Cond);
                            for sub=1:length(subnums)
                                minfrac=prctile(TestData2.(T1difftxt)(TestData2.sub==num2str(subnums(sub))),100*(ii-1)/fractions);
                                maxfrac=prctile(TestData2.(T1difftxt)(TestData2.sub==num2str(subnums(sub))),100*ii/fractions);
                                if length(unique(TestData2.(T1difftxt)))==2 % 0 or 1 binary result
                                    if ii==1
                                        minfrac=0;maxfrac=0.5;
                                    elseif ii==2
                                        minfrac=0.5;maxfrac=1;
                                    else
                                        minfrac=NaN;maxfrac=NaN;
                                    end
                                end
                                TestData2((TestData2.(T1difftxt)<minfrac | TestData2.(T1difftxt)>maxfrac) & TestData2.sub==num2str(subnums(sub)),:)=[];
                            end
                            disp([T2T1benefittxt,'~Cond+(Cond|sub):  part ',num2str(ii),'/',num2str(fractions)]);
                            lm=fitglme(TestData2,[T2T1benefittxt,'~Cond+(Cond|sub)']);
                            disp(anova(lm))%draw_lm(lm);
                        end
                    end
                catch err
                    disp('Analysis failed');
                end
                %%
                try
                    if ~isempty(T2difftxt)
                        disp(['Splitting the Data by T1 results (default: median split) and considering the two halves as a factor:',add_text_Cond])
                        fractions=2;
                        TestData2=TestData;
                        TestData2.MemFraction(:,1)=nan;
                        TestData2.Cond(TestData2.Cond==4)=3;TestData2.Cond=ordinal(TestData2.Cond);
                        for ii=1:fractions
                            for sub=1:length(subnums)
                                minfrac=prctile(TestData2.(T1difftxt)(TestData2.sub==num2str(subnums(sub))),100*(ii-1)/fractions);
                                maxfrac=prctile(TestData2.(T1difftxt)(TestData2.sub==num2str(subnums(sub))),100*ii/fractions);
                                if length(unique(TestData2.(T1difftxt)))==2
                                    if ii==1
                                        minfrac=0;maxfrac=0.5;
                                    elseif ii==2
                                        minfrac=0.5;maxfrac=1;
                                    else
                                        minfrac=NaN;maxfrac=NaN;
                                    end
                                end
                                TestData2.MemFraction(TestData2.(T1difftxt)>=minfrac & TestData2.(T1difftxt)<=maxfrac & TestData2.sub==num2str(subnums(sub)),:)=ii;
                            end
                        end
                        TestData2.MemFraction=categorical(TestData2.MemFraction);
                        disp([T2T1benefittxt,'~MemFraction*Cond+(MemFraction*Cond|sub)']);
                        lm=fitglme(TestData2,[T2T1benefittxt,'~MemFraction*Cond+(MemFraction*Cond|sub)']);
                        disp(anova(lm))%draw_lm(lm);
                    end
                catch err
                    disp('Analysis failed');
                end                
                
%% Distance analyses

%% Analyses to display

T2difftxt='T2DiffIfClosest';
T1difftxt='DiffIfClosest';
TestData2Disp{1}=TestData;
TestData2Disp{2}=TestData;
TestData2Disp{1}.Cond=categorical(TestData2Disp{1}.Cond);
TestData2Disp{2}.Cond=categorical(TestData2Disp{2}.Cond);
for sub=1:length(subnums)
    T1mean=nanmean(TestData2Disp{1}.(T1difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub))));
    T1std=nanstd(TestData2Disp{1}.(T1difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub))));
    TestData2Disp{1}.(T1difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub)))=(TestData2Disp{1}.(T1difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub)))-T1mean)./T1std;
    TestData2Disp{1}.(T2difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub)))=(TestData2Disp{1}.(T2difftxt)(TestData2Disp{1}.sub==num2str(subnums(sub)))-T1mean)./T1std;
end
for ii=[2,1]
    TestData2Disp{ii}.Cond=mergecats(TestData2Disp{ii}.Cond,["1","2"]);
    % TestData2Disp.Cond=removecats(TestData2Disp.Cond,"2");
    TestData2Disp{ii}.Cond=renamecats(TestData2Disp{ii}.Cond,"3","2");
    TestData2Disp{ii}.Cond=renamecats(TestData2Disp{ii}.Cond,"4","3");
    lm1=fitglme(TestData2Disp{ii},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
    figure;set(gcf,'windowstyle','normal','units','normalized','position',[0.0958    0.2100    0.7557    0.4092])
    subplot(1,3,2);hold all;
    [dat,x_lab]=draw_lm(lm1);
    range=[min(floor([dat(:);x_lab{1}']*2)),max(ceil([dat(:);x_lab{1}']*2))]/2;
    range(2)=5.25;
    if ii==2
        range=[0 270];
    end
    line([0 0],range,'color',[1 1 1]*0.8);
    line(range,[0 0],'color',[1 1 1]*0.8);
    a=plot(repmat(x_lab{1},[size(dat,1),1])',dat','linewidth',3);
    clrs=[a.Color];clrs=reshape(clrs,[size(dat,1),3]);

    xlim(range);ylim(range);
    if ii==1
        set(gca,'xtick',-2:2:4,'ytick',-2:2:4,'FontSize',14)
    else
        set(gca,'xtick',range,'ytick',range,'FontSize',14)
    end
    lg=legend(a(end:-1:1),['Cued objects ',char(8712),' cued set'],['Non-cued objects ',char(8712),' cued set'],['Non-cued objects ',char(8713),' cued set']);
    lg.EdgeColor='w';
    lg.FontSize=10;
    lg.Location='North';
    if ii==2
        xlabel('Pre-sleep error (pixels)');
        ylabel('Post-sleep error (pixels)');
    else
        xlabel('Pre-sleep error (Z-score)');
        ylabel('Post-sleep error (Z-score)');
    end
    title('All participants');

    disp(anova(lm1))%draw_lm(lm);

    TestData2Disp{ii}.Cond=reordercats(TestData2Disp{ii}.Cond,["2","1",'3']);
    lm2=fitglme(TestData2Disp{ii},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
    TestData2Disp{ii}.Cond=reordercats(TestData2Disp{ii}.Cond,["3","1",'2']);
    lm3=fitglme(TestData2Disp{ii},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
    subplot(1,3,3);hold all;
    bar(1:3,[lm3.Coefficients.Estimate(2),nan,nan],'FaceColor',clrs(:,3),'EdgeColor',clrs(:,3));line([1 1],lm3.Coefficients.Estimate(2)+[0 lm3.Coefficients.SE(2)],'color',clrs(:,3),'linewidth',2);
    bar(1:3,[nan,lm2.Coefficients.Estimate(2),nan],'FaceColor',clrs(:,2),'EdgeColor',clrs(:,2));line([2 2],lm2.Coefficients.Estimate(2)+[0 lm2.Coefficients.SE(2)],'color',clrs(:,2),'linewidth',2);
    bar(1:3,[nan,nan,lm1.Coefficients.Estimate(2)],'FaceColor',clrs(:,1),'EdgeColor',clrs(:,1));line([3 3],lm1.Coefficients.Estimate(2)+[0 lm1.Coefficients.SE(2)],'color',clrs(:,1),'linewidth',2);
    ylabel('\fontsize{15.4}{Encoding-strength-dependent forgetting}\fontsize{12}\newline      Less forgetting <---------> More Forgetting');% % % $ %
    line([2 3],max([lm1.Coefficients.Estimate(2) lm2.Coefficients.Estimate(2)])+[1 1]*max([lm1.Coefficients.SE(2),lm2.Coefficients.SE(2)]),'color','k');
    shift=0.02;if lm1.Coefficients.pValue(5)<0.001,            ptext='***';        elseif lm1.Coefficients.pValue(5)<0.01,            ptext='**';        elseif lm1.Coefficients.pValue(5)<0.05,            ptext='*';         elseif lm1.Coefficients.pValue(5)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
    text(2.5,max([lm1.Coefficients.Estimate(2) lm2.Coefficients.Estimate(2)])+max([lm1.Coefficients.SE(2),lm2.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
    line([1 3],max([lm1.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+[1 1]*1.4*max([lm1.Coefficients.SE(2),lm3.Coefficients.SE(2)]),'color','k');
    shift=0.02;if lm1.Coefficients.pValue(6)<0.001,            ptext='***';        elseif lm1.Coefficients.pValue(6)<0.01,            ptext='**';        elseif lm1.Coefficients.pValue(6)<0.05,            ptext='*';         elseif lm1.Coefficients.pValue(6)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
    text(2,max([lm1.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+1.4*max([lm1.Coefficients.SE(2),lm3.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
    line([1 2],max([lm2.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+[1 1]*1.5*max([lm2.Coefficients.SE(2),lm3.Coefficients.SE(2)]),'color','k');
    shift=0.02;if lm2.Coefficients.pValue(6)<0.001,            ptext='***';        elseif lm2.Coefficients.pValue(6)<0.01,            ptext='**';        elseif lm2.Coefficients.pValue(6)<0.05,            ptext='*';         elseif lm2.Coefficients.pValue(6)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
    text(1.5,max([lm2.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+1.5*max([lm2.Coefficients.SE(2),lm3.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
    set(gca,'xtick',[],'ytick',[0 0.4 0.8],'fontsize',15);
    ylim([0 0.9])

    % Data per sub
    TestData2Disp{ii}.Cond=reordercats(TestData2Disp{ii}.Cond,["1","2","3"]);
    subdata=nan(length(subnums),5);
    for sub=1:length(subnums)
        if subnums(sub)~=469 && exist('Subdata','file')
            continue;
        end
        TestData2DispSUB=TestData2Disp{ii}(TestData2Disp{ii}.sub==num2str(subnums(sub)),:);
        lmSUB=fitglme(TestData2DispSUB,[T2difftxt,'~',T1difftxt,'*Cond']);
        
        if ii==1
            lmSUB2=fitglme(TestData2DispSUB,[T2difftxt,'~',T1difftxt]);
            TestData2DispSUB2=TestData2DispSUB;
            TestData2DispSUB2.Cond=reordercats(TestData2DispSUB2.Cond,["3","2","1"]);
            lmSUB3=fitglme(TestData2DispSUB2,[T2difftxt,'~',T1difftxt,'*Cond']);
            subdata(sub,:)=[lmSUB.Coefficients.Estimate(5),lmSUB.Coefficients.Estimate(6),lmSUB.Coefficients.Estimate(2),lmSUB2.Coefficients.Estimate(2),lmSUB3.Coefficients.Estimate(6)];
        end
        % First column is the effect on the non-cued images in a cued set, second is the effect on cued images 
        if subnums(sub)==469
            subplot(1,3,1);
            [dat,x_lab]=draw_lm(lmSUB);
            hold all;
            line([0 0],range,'color',[1 1 1]*0.8);
            line(range,[0 0],'color',[1 1 1]*0.8);
            a=plot(repmat(x_lab{1},[size(dat,1),1])',dat','linewidth',3);
            clrs=[a.Color];clrs=reshape(clrs,[size(dat,1),3]);
            hold all;
            for jj=1:3,hold all;scatter(TestData2DispSUB.(T1difftxt)(TestData2DispSUB.Cond==num2str(jj)),TestData2DispSUB.(T2difftxt)(TestData2DispSUB.Cond==num2str(jj)),'MarkerEdgeColor',clrs(:,jj),'MarkerFaceColor',clrs(:,jj));end;
            xlim(range);ylim(range);
            title(['Participant #',num2str(subnums(sub))]);
            if ii==1
                xlabel('Pre-sleep error (Z-score)');
                ylabel('Post-sleep error (Z-score)');
                set(gca,'xtick',-2:2:4,'ytick',-2:2:4,'FontSize',14)
            else
                xlabel('Pre-sleep error (pixels)');
                ylabel('Post-sleep error (pixels)');
                set(gca,'xtick',range,'ytick',range,'FontSize',14)
            end
        end

    end
    if ii==1
        annotation('arrow',[0.206797381116471,0.228118538938663],[0.656651583710407,0.791855203619909],'color',[1 1 1]*0.7)
        annotation('arrow',[0.263955892487939,0.306838042729152],[0.382352941176471,0.395063348416289],'color',[1 1 1]*0.7)
        text(0.1875,3.12,{'    More','Forgetting'},'fontsize',8,'rotation',64,'color',[1 1 1]*0.7)
        text(2.48,0.6275,{'    Less','Forgetting'},'fontsize',8,'rotation',6,'color',[1 1 1]*0.7)
    end
    save('Subdata','subdata','subnums','TestData');
end
%% Same but including temporal context
T2difftxt='T2DiffIfClosest';
T1difftxt='DiffIfClosest';
TestData2Disp{3}=TestData;
TestData2Disp{3}.Cond=categorical(TestData2Disp{3}.Cond);
for sub=1:length(subnums)
    T1mean=nanmean(TestData2Disp{3}.(T1difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub))));
    T1std=nanstd(TestData2Disp{3}.(T1difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub))));
    TestData2Disp{3}.(T1difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub)))=(TestData2Disp{3}.(T1difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub)))-T1mean)./T1std;
    TestData2Disp{3}.(T2difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub)))=(TestData2Disp{3}.(T2difftxt)(TestData2Disp{3}.sub==num2str(subnums(sub)))-T1mean)./T1std;
end
lm1=fitglme(TestData2Disp{3},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
[dat,x_lab]=draw_lm(lm1);range=[min(floor([dat(:);x_lab{1}']*2)),max(ceil([dat(:);x_lab{1}']*2))]/2;
range(2)=5.25;
figure;set(gcf,'windowstyle','normal','units','normalized','position',[0.0958    0.2100    0.7557    0.4092])
subplot(1,3,2);hold all;
line([0 0],range,'color',[1 1 1]*0.8);
line(range,[0 0],'color',[1 1 1]*0.8);
line(x_lab{1},dat(1,:),'color',[0 218 210]/255,'linewidth',3);
a=plot(repmat(x_lab{1},[size(dat,1),1])',dat','linewidth',3)
tmp=a(4).Color;a(4).Color=a(3).Color;a(3).Color=a(2).Color;a(2).Color=tmp;
% a(1).Color=[0,115,74]/255;
a(1).Color='w';
a(2).Color=[0 0 150]/255;
a(1).LineStyle=':';
a(2).LineStyle=':';
clrs=[a.Color];clrs=reshape(clrs',[3,size(dat,1)])';clrs(1,:)=[0 218 210]/255;

xlim(range);ylim(range);
lg=legend(a(end:-1:1),['Cued objects ',char(8713),' cued set'],['Non-cued objects ',char(8712),' cued set'],['Non-cued objects ',char(8712),' cued block'],['Non-cued objects ',char(8713),' cued block'])
set(gca,'xtick',-2:2:4,'ytick',-2:2:4,'FontSize',14)
lg.EdgeColor='w';
lg.FontSize=10;
lg.Location='North';
xlabel('Pre-sleep error (Z-score)');
ylabel('Post-sleep error (Z-score)');
title('All participants');

disp(anova(lm1))

TestData2Disp{3}.Cond=reordercats(TestData2Disp{3}.Cond,["2","1","3","4"]);
lm2=fitglme(TestData2Disp{3},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
TestData2Disp{3}.Cond=reordercats(TestData2Disp{3}.Cond,["3","1","2","4"]);
lm3=fitglme(TestData2Disp{3},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);
TestData2Disp{3}.Cond=reordercats(TestData2Disp{3}.Cond,["4","1","2","3"]);
lm4=fitglme(TestData2Disp{3},[T2difftxt,'~',T1difftxt,'*Cond+(',T1difftxt,'*Cond|sub)']);

subplot(1,3,3);hold all;
bar(1:4,[lm4.Coefficients.Estimate(2),nan,nan,nan],'FaceColor',clrs(4,:),'EdgeColor',clrs(4,:));line([1 1],lm4.Coefficients.Estimate(2)+[0 lm4.Coefficients.SE(2)],'color',clrs(4,:),'linewidth',2);
bar(1:4,[nan,lm3.Coefficients.Estimate(2),nan,nan],'FaceColor',clrs(3,:),'EdgeColor',clrs(3,:));line([2 2],lm3.Coefficients.Estimate(2)+[0 lm3.Coefficients.SE(2)],'color',clrs(3,:),'linewidth',2);
bar(1:4,[nan,nan,lm2.Coefficients.Estimate(2),nan],'FaceColor',clrs(2,:),'EdgeColor',clrs(2,:));line([3 3],lm2.Coefficients.Estimate(2)+[0 lm2.Coefficients.SE(2)],'color',clrs(2,:),'linewidth',2);
bar(1:4,[nan,nan,nan,lm1.Coefficients.Estimate(2)],'FaceColor',clrs(1,:),'EdgeColor',clrs(1,:));line([4 4],lm1.Coefficients.Estimate(2)+[0 lm1.Coefficients.SE(2)],'color',clrs(1,:),'linewidth',2);
ylabel('\fontsize{15.4}{Encoding-strength-dependent forgetting}\fontsize{12}\newline      Less forgetting <---------> More Forgetting');% % % $ %
line([3 4],max([lm1.Coefficients.Estimate(2) lm2.Coefficients.Estimate(2)])+[1 1]*1.2*max([lm1.Coefficients.SE(2),lm2.Coefficients.SE(2)]),'color','k');
shift=0.02;if lm1.Coefficients.pValue(6)<0.001,            ptext='***';        elseif lm1.Coefficients.pValue(6)<0.01,            ptext='**';        elseif lm1.Coefficients.pValue(6)<0.05,            ptext='*';         elseif lm1.Coefficients.pValue(6)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
text(3.5,max([lm1.Coefficients.Estimate(2) lm2.Coefficients.Estimate(2)])+1.2*max([lm1.Coefficients.SE(2),lm2.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
line([2 4],max([lm1.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+[1 1]*2*max([lm1.Coefficients.SE(2),lm3.Coefficients.SE(2)]),'color','k');
shift=0.02;if lm1.Coefficients.pValue(7)<0.001,            ptext='***';        elseif lm1.Coefficients.pValue(7)<0.01,            ptext='**';        elseif lm1.Coefficients.pValue(7)<0.05,            ptext='*';         elseif lm1.Coefficients.pValue(7)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
text(3,max([lm1.Coefficients.Estimate(2) lm3.Coefficients.Estimate(2)])+2*max([lm1.Coefficients.SE(2),lm3.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
line([1 4],max([lm1.Coefficients.Estimate(2) lm4.Coefficients.Estimate(2)])+[1 1]*2.8*max([lm1.Coefficients.SE(2),lm4.Coefficients.SE(2)]),'color','k');
shift=0.02;if lm1.Coefficients.pValue(8)<0.001,            ptext='***';        elseif lm1.Coefficients.pValue(8)<0.01,            ptext='**';        elseif lm1.Coefficients.pValue(8)<0.05,            ptext='*';         elseif lm1.Coefficients.pValue(8)<0.1,            ptext='~';shift=0;         else,            ptext='n.s';shift=0;        end
text(2.5,max([lm1.Coefficients.Estimate(2) lm4.Coefficients.Estimate(2)])+2.8*max([lm1.Coefficients.SE(2),lm4.Coefficients.SE(2)])-shift,ptext,'verticalalignment','bottom','horizontalalignment','center','fontsize',12);
set(gca,'xtick',[],'ytick',[0 0.4 0.8],'fontsize',15);
ylim([0 0.9])
print(gcf,'foo2.png','-dpng','-r1200');