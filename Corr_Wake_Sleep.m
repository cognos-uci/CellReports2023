
try
    clear;
    all_trials_vs_shown_later_vs_not_shown_late=3;
    load([pwd,'\Datasets\sublists']);
    subnums=subs_atleast3itemrecall;
    subnums=setdiff(subnums,[431,432]);%participants with no pre-sleep classification test
    
    tic,load([pwd,'\EEG\',num2str(subnums(1)),'a.mat'],'data_trial')
    toc
    tic,load([pwd,'\EEG\',num2str(subnums(1)),'a.mat'],'data','trl','cfg_art_rej')
    toc
    pretime=1.49;posttime=4.99;
    interval=11;
    tois=find(abs(data_trial.time{1}-(-pretime))==min(abs(data_trial.time{1}-(-pretime)))):interval:find(abs(data_trial.time{1}-(posttime))==min(abs(data_trial.time{1}-(posttime))));
    TOI_indices=bsxfun(@plus,tois',-floor((interval-1)/2):ceil((interval-1)/2));
    TOI_indices(TOI_indices<1 | TOI_indices>length(data_trial.time{1}))=NaN;
    real_tois=data_trial.time{1}(tois);
    filename=['Dataset',num2str(size(ls('Dataset*.mat'),1)+1),'.mat'];

    m_file_text=fileread(matlab.desktop.editor.getActiveFilename);
    m_file_name=matlab.desktop.editor.getActiveFilename;
    cfg=[];
    cfg.hpfilter='yes';
    cfg.hpfreq=14;
    cfg.lpfilter='yes';
    cfg.lpfreq=18;
    cfg.padding=10;

    for sub = 1:length(subnums)
        disp(['Subject #',num2str(sub),'/',num2str(length(subnums))]);
        load([pwd,'\EEG\',num2str(subnums(sub)),'a.mat'],'data_trial','WasNotContextImage','WasContextImage','WhichImagePresented')
        dataA=data_trial;
        clearvars data_trial;
        load([pwd,'\EEG\',num2str(subnums(sub)),'s.mat'],'data_trial')
        dataS=data_trial;
        if length(dataA.label)~=length(dataS.label)
            error('Bad labels');
        end
        [~,a2,a3]=intersect(dataA.label,dataS.label);
        dataA.label=dataA.label(a2);
        for trial=1:length(dataA.trial)
            dataA.trial{trial}=dataA.trial{trial}(a2,:);
        end
        dataS.label=dataS.label(a3);
        for trial=1:length(dataS.trial)
            dataS.trial{trial}=dataS.trial{trial}(a3,:);
        end
        clearvars data_trial;
        dataA.trialinfo(:,1)=mod(dataA.trialinfo(:,1),10);
        loc_trials=find(dataA.cfg.trlold(:,4)==21);
        if all_trials_vs_shown_later_vs_not_shown_late==2
            dataA.trialinfo(find(ismember(dataA.sampleinfo(:,1),dataA.cfg.trlold(loc_trials(WasNotContextImage),1))),1)=4;
        elseif all_trials_vs_shown_later_vs_not_shown_late==3
            dataA.trialinfo(find(ismember(dataA.sampleinfo(:,1),dataA.cfg.trlold(loc_trials(WasContextImage),1))),1)=4;
        end
        gmail(num2str(sub));
        trialinfo=dataA.trialinfo;
        did_change_in_trial_info=0;
        dataA.trialinfo=trialinfo;
        svmECOC{sub,4}.nBins=2;                          
        [a,~,c]=unique(dataA.trialinfo(:,1));
        svmECOC{sub,4}.nIter = 20; % # of iterations
        svmECOC{sub,4}.nBlocks = 5; % # of blocks for cross-validation
        svmECOC{sub,4}.Fs = dataA.fsample; % samplring rate of in the preprocessed dataA for filtering
        tm=dataA.time{1}(tois);
        nPerBin=hist(dataA.trialinfo(dataA.trialinfo(:,1)<=svmECOC{sub,4}.nBins,1),1:svmECOC{sub,4}.nBins);
        svmECOC{sub,4}.nTrials=sum(nPerBin);
        minCnt = min(nPerBin);
        nBlocks=svmECOC{sub,4}.nBlocks;
        nBins=svmECOC{sub,4}.nBins;
        Elecs=1:64;
        nElectrodes=length(Elecs);
        nSamps=length(tois);
        Nitr=svmECOC{sub,4}.nIter;
        if ~did_change_in_trial_info
            did_change_in_trial_info=1;
            dataS.trialinfo(:,2)=dataS.trialinfo(:,1);
            idxs=find(dataS.trialinfo(:,2)~=76);
            dataS.trialinfo(idxs,2)=1;
        end
        SleepBins=1;
        sleep_eeg=zeros(length(Elecs),length(tois),length(SleepBins));
        for bin=1:length(SleepBins)
            tmp_sleep_eeg=cell2mat(reshape(dataS.trial(dataS.trialinfo(:,2)==SleepBins(bin)),[1,1,numel(dataS.trial(dataS.trialinfo(:,2)==SleepBins(bin)))]));
            tmp_sleep_eeg_ave=mean(tmp_sleep_eeg(Elecs,:,:),3);
            sleep_trial_eeg{bin}=zeros(length(Elecs),length(TOI_indices),size(tmp_sleep_eeg,3));
            for elec=1:length(Elecs)
                temp=tmp_sleep_eeg_ave(elec,:);
                sleep_eeg(elec,:,bin)=nanmean(temp(TOI_indices),2);
                for trial=1:size(sleep_trial_eeg{bin},3)
                    sleep_trial_eeg{bin}(elec,:,trial)=nanmean(reshape(tmp_sleep_eeg(elec,TOI_indices',trial),[size(TOI_indices,2),size(TOI_indices,1)])',2)';
                end
            end
            clearvars tmp_sleep_eeg_ave tmp_sleep_eeg temp;
            
        end
        PerSoundSleepBins=unique(dataS.trialinfo(:,1));PerSoundSleepBins=setdiff(PerSoundSleepBins,76);
        svmECOC{sub,4}.modelPredictSleep=nan(Nitr,nSamps,nSamps,length(SleepBins));
        svmECOC{sub,4}.targetsSleep=nan(Nitr,nSamps,nSamps,length(SleepBins));
        svmECOC{sub,4}.modelPredictSleep_persound=nan(Nitr,nSamps,nSamps,length(PerSoundSleepBins));
        svmECOC{sub,4}.targetsSleep_persound=nan(Nitr,nSamps,nSamps,length(PerSoundSleepBins));
        for iter = 1:Nitr
            disp(['iter=',num2str(iter),'/',num2str(Nitr)]);
            blockDat_filtData = nan(nBins*nBlocks,nElectrodes,nSamps);   
            labels = nan(nBins*nBlocks,1);     % bin labels for averaged & filtered EEG dataA
            blockNum = nan(nBins*nBlocks,1);   % block numbers for averaged & filtered EEG dataA
            bCnt = 1; %initialize binBlocks counter 
            for bin = 1:nBins 
                nPerBinBlocks = floor(minCnt/nBlocks);
                shuffBin = randperm((nPerBinBlocks*nBlocks))';
                blocks = nan(size(shuffBin));
                shuffBlocks = nan(size(shuffBin));
                y = repmat((1:nBlocks)',nPerBinBlocks,1);
                shuffBlocks(shuffBin) = y;
                blocks(shuffBin) = shuffBlocks;
                eeg_now=cell2mat(reshape(dataA.trial(dataA.trialinfo(:,1)==bin),[1,1,numel(dataA.trial(dataA.trialinfo(:,1)==bin))]));
                eeg_now=eeg_now(Elecs,:,:);
                if iter==1
                    wake_trial_eeg=zeros(length(Elecs),length(TOI_indices),size(eeg_now,3));
                    for elec=1:length(Elecs)
                        for trial=1:size(eeg_now,3)
                            wake_trial_eeg(elec,:,trial)=nanmean(reshape(eeg_now(elec,TOI_indices',trial),[size(TOI_indices,2),size(TOI_indices,1)])',2)';
                        end
                    end

                    eeg_now_ave=nanmean(wake_trial_eeg,3);
                    for trial=1:size(sleep_trial_eeg{1},3)
                        for ii=1:size(TOI_indices,1)
                            for jj=1:size(TOI_indices,1)
                                svmECOC{sub,4}.TrialCorr(ii,jj,trial,bin)=corr(eeg_now_ave(:,ii),sleep_trial_eeg{1}(:,jj,trial));
                            end
                        end
                    end
                    clearvars eeg_now_ave;
                end
                for bl = 1:nBlocks
                    tmp_sleep_eeg=mean(eeg_now(:,:,blocks==bl),3);
                    for elec=1:length(Elecs)
                        temp=tmp_sleep_eeg(elec,:);
                        blockDat_filtData(bCnt,elec,:)=nanmean(temp(TOI_indices),2);
                    end                       
                    clearvars tmp_recall_eeg temp;
                    
                    labels(bCnt) = bin; %used for arranging classification obj.

                    blockNum(bCnt) = bl;

                    bCnt = bCnt+1;

                end
            end
            for t = 1:nSamps
                toi = t;
                dataAtTimeT = squeeze(mean(blockDat_filtData(:,:,toi),3));
                for i=1:nBlocks 
                    trnl = labels(blockNum~=i); % training labels
                    tstl = labels(blockNum==i); % test labels
                    trnD = dataAtTimeT(blockNum~=i,:);    % training dataA
                    tstD = dataAtTimeT(blockNum==i,:);    % test dataA
                    mdl = fitcecoc(trnD,trnl, 'Coding','onevsall','Learners','SVM' );   %train support vector mahcine
                    % Step 5: Testing
                    LabelPredicted = predict(mdl, tstD);       % predict classes for new dataA

                    svmECOC{sub,4}.modelPredict(iter,t,i,:) = LabelPredicted;  % save predicted labels

                    svmECOC{sub,4}.targets(iter,t,i,:) = tstl;             % save true target labels


                end % end of block: Step 6: cross-validation
                if is_sleep && iter==1

                    % Step 2 & Step 4: Assigning training and testing dataA sets
                    trnl = labels; % training labels
                    trnD = dataAtTimeT;    % training dataA
                    tstl = 1; % test labels
                    mdl = fitcecoc(trnD,trnl, 'Coding','onevsall','Learners','SVM' );   %train support vector mahcine
                    clearvars c;
                    for ii=1:size(sleep_eeg,2)
                        for jj=1:size(sleep_eeg,3)
                            c(ii,jj)=corr(sleep_eeg(:,ii,jj),nanmean(trnD(trnl==1,:),1)'); %correlation of sleep data with category 1, which is the place
                        end
                    end
                    svmECOC{sub,4}.RSAres1(iter,t,:,:)=c;
                    for ii=1:size(sleep_eeg,2)
                        for jj=1:size(sleep_eeg,3)
                            c(ii,jj)=corr(sleep_eeg(:,ii,jj),nanmean(trnD(trnl==2,:),1)'); %correlation of sleep data with category 2, which is the abstract image
                        end
                    end
                    svmECOC{sub,4}.RSAres2(iter,t,:,:)=c;
                end

            end % end of time points: Step 7: Decoding each time point
        end % end of iteration: Step 8: iteration with random shuffling
        svmECOC{sub,4}.nBlocks = nBlocks;

        DecodingAccuracy = nan(nSamps,nBlocks,Nitr);
        % Obtain predictions from SVM-ECOC model
        svmPrediction = (svmECOC{sub,4}.modelPredict);
        tstTargets = (svmECOC{sub,4}.targets);
        %clear svmECOC{sub,4} (is now a large unnecessary objet)

        % Step 5: Compute decoding accuracy of each decoding trial
        for block = 1:size(svmPrediction,3)
            for itr = 1:Nitr
                for tp = 1:nSamps  

                    prediction = squeeze(svmPrediction(itr,tp,block,:)); % this is predictions from models
                    TrueAnswer = squeeze(tstTargets(itr,tp,block,:)); % this is predictions from models
                    Err = TrueAnswer - prediction; %compute error. No error = 0
                    ACC = mean(Err==0); %Correct hit = 0 (avg propotion of vector of 1s and 0s)
                    DecodingAccuracy(tp,block,itr) = ACC; % average decoding accuracy at tp & block

                end
            end
        end

         % Average across block and iterations
        grandAvg = squeeze(mean(mean(DecodingAccuracy,2),3));

         % Perform temporal smoothing (7 point moving avg) 
        smoothwin=7;%
        smoothed = nan(1,nSamps+2*floor(smoothwin/2));
        grandAvgtmp = nan(1,nSamps+2*floor(smoothwin/2));
        grandAvgtmp(floor(smoothwin/2)+1:nSamps+floor(smoothwin/2))=grandAvg;
        for tAvg = floor(smoothwin/2)+1:nSamps+floor(smoothwin/2)
          smoothed(tAvg) = nanmean(grandAvgtmp((tAvg-floor(smoothwin/2)):(tAvg+floor(smoothwin/2))));  
        end
        smoothed=smoothed(floor(smoothwin/2)+1:nSamps+floor(smoothwin/2));

         % Save smoothe dataA
        AverageAccuracy(sub,:,4) =smoothed; % average across iteration and block          
        clearvars dataA dataS eeg_now tstTargets grandAvg sleep_trial_eeg;
        
        
    end %End of subject
    save(filename,'-v7.3');
    %% WAKE
    subAverage = squeeze(mean(AverageAccuracy,1)); 
    seAverage = squeeze(std(AverageAccuracy,1))/sqrt(length(subnums)); 

    % Visualization:Plotting the decoding accuracy 
    % across all subjects at each timepoint
    % Not publication quality
    figure;
    subplot(1,size(subAverage,2),4);
    try
        chancelvl = 1/svmECOC{1,4}.nBins; %chance level of avg. decoding
    catch
        continue;
    end
            
    tm=real_tois;
    cl=colormap(parula(50));
    plot(tm, subAverage(:,4)); %plot
    boundedline(tm,subAverage(:,4),seAverage(:,4),'cmap',cl(42,:),'alpha','transparency',0.35)
    line([tm(1),tm(length(tm))],[chancelvl,chancelvl]); %chance line
    switch 4
        case 1
            title('Places vs Faces vs Abstract');
        case 2
            title('Places vs Abstract');                        
        case 3
            title('Faces vs Abstract');
        case 4
            title('Places vs Faces');
    end    
    limit=get(gca,'ylim');
    line([0 0],[0 1]);
    limit=set(gca,'ylim',limit);
         
    
    
    %% SLEEP
    clearvars dataRSA ttestRSA smootheddataRSA smoothwin smootheddataRSA_TMP;
    cl=colormap(parula(50));
    subAverage = squeeze(mean(AverageAccuracy,1)); 
    seAverage = squeeze(std(AverageAccuracy,1))/sqrt(length(subnums)); 
    smoothwin=5;%odd
    for sub=1:length(subnums)
        dataRSA(:,:,sub,1)=squeeze(svmECOC{sub,4}.RSAres1(1,:,:,1));
        dataRSA(:,:,sub,2)=squeeze(svmECOC{sub,4}.RSAres2(1,:,:,1));
    end
    
    smootheddataRSA=nan(size(dataRSA,1)+smoothwin-1,size(dataRSA,2)+smoothwin-1,size(dataRSA,3),size(dataRSA,4));
    smootheddataRSA(ceil(smoothwin/2):end-floor(smoothwin/2),ceil(smoothwin/2):end-floor(smoothwin/2),:,:)=dataRSA;
    smootheddataRSA_TMP=smootheddataRSA;
    for ii=ceil(smoothwin/2):size(smootheddataRSA,1)-floor(smoothwin/2)
        for jj=ceil(smoothwin/2):size(smootheddataRSA,2)-floor(smoothwin/2)
            smootheddataRSA(ii,jj,:,:)=nanmean(nanmean(smootheddataRSA_TMP(ii-floor(smoothwin/2):ii+floor(smoothwin/2),jj-floor(smoothwin/2):jj+floor(smoothwin/2),:,:),1),2);
        end
    end
    smootheddataRSA=smootheddataRSA(ceil(smoothwin/2):end-floor(smoothwin/2),ceil(smoothwin/2):end-floor(smoothwin/2),:,:);
    smootheddataRSA(:,:,:,3)=smootheddataRSA(:,:,:,1)-smootheddataRSA(:,:,:,2);
    smootheddataRSA(:,:,:,4)=smootheddataRSA(:,:,:,3)-repmat(nanmean(smootheddataRSA(:,1:find(tm<=0,1,'last'),:,3),2),[1,size(smootheddataRSA,1) 1,1]);
    
    cl=colormap(parula(50));
    subAverage = squeeze(mean(AverageAccuracy,1)); 
    seAverage = squeeze(std(AverageAccuracy,1))/sqrt(length(subnums)); 
    
    figure;
    set(gcf,'windowstyle','normal','windowstate','maximized');
    limy=[-1.5 3];
    limy_for_analysis=limy;
    limx=[-0.75 2];
    cbar=[0.8 1.9;2.15 2.6];    
    clrs=[0 0 1;1 0 0;0 1 0];
    subplot(24,4,1:4:25);
    4=4;
    chancelvl = 1/svmECOC{1,4}.nBins; %chance level of avg. decoding
    tm=real_tois;
    plot(tm, subAverage(:,4)); %plot
    boundedline(tm,subAverage(:,4),seAverage(:,4),'cmap',cl(42,:),'alpha','transparency',0.35)
    line([tm(1),tm(length(tm))],[chancelvl,chancelvl]); %chance line
    xl=xlabel('WAKE: Time relative to image onset (s)');
    set(xl,'position',[mean(limx) 0.428 -1]);
    ylabel('Classification accuracy');

    limit=get(gca,'ylim');
    line([0 0],[0 1]);
    set(gca,'ylim',limit,'xlim',limx,'fontsize',12,'xtick',[-0.5 1.5 2],'xticklabel',{'\color{black}-0.5','\color{black}1.5','\color{black}2'});
    axes('position',[0.13 0.6541 0.1566 0.04],'xtick',[],'ytick',[]);hold all;fill([0 0 1 1],[0 1 1 0],'k','edgecolor','none');text(0.5,0.5,'Place','color',[1 1 1]*0.999,'horizontalalignment','center','verticalalignment','middle');fill([0 0 1 1],1+[0 1 1 0],[1 1 1]*0.8,'edgecolor','none');text(0.5,1.5,'Abstract','color','k','horizontalalignment','center','verticalalignment','middle');axis off;set(gca,'xlim',limx,'ylim',[0 2])
       
    for ii=1:2
        subplot(24,4,(41:4:61)+(ii==2)*28);
        imagesc(tm,tm,nanmean(smootheddataRSA(:,:,:,ii),3)');
        set(gca,'clim',[-0.3 0.1],'ydir','normal','xlim',limx,'ylim',limy);
        set(gca,'fontsize',12,'ydir','normal','ticklength',[0.03 0],'ticklength',[0.03 0],'xtick',[-1:0.5:2.5],'xticklabel',{'\color{black}-1','\color{black}-0.5','','','','\color{black}1.5','\color{black}2','\color{black}2.5'},'ytick',-1:2,'yticklabel',{'-1',char([55357,56586]),'1','2'});
        clrmap=colormap(gca);
        line([0 0],limy,'color','k');
        line(limx,[0 0],'color','k');
        hold all;
        fill([cbar(1,1) cbar(1,2) cbar(1,2) cbar(1,1)],[cbar(2,1) cbar(2,1) cbar(2,2) cbar(2,2)],'w');
        for jj=1:size(clrmap,1)
            fill(cbar(1,1)+diff(cbar(1,:))*([0 1 1 0]+jj-1)/size(clrmap,1),[cbar(2,1) cbar(2,1) cbar(2,2) cbar(2,2)],clrmap(jj,:),'edgecolor','none');
        end
        text(cbar(1,1)+0.02,mean(cbar(2,:),2),'-0.3','horizontalalignment','left','verticalalignment','middle','fontsize',10,'color','w');
        text(mean(cbar(1,:)),mean(cbar(2,:),2),'{\itr}','horizontalalignment','center','verticalalignment','middle','fontsize',12,'color','w');
        text(cbar(1,2)-0.02,mean(cbar(2,:),2),'0.1','horizontalalignment','right','verticalalignment','middle','fontsize',10);
        if ii==1
            axes('position',[0.13 0.3649 0.1566 0.02],'xtick',[],'ytick',[]);hold all;fill([0 0 1 1],[0 1 1 0],'k','edgecolor','none');text(0.5,0.5,'Place','color',[1 1 1]*0.999,'horizontalalignment','center','verticalalignment','middle');axis off;set(gca,'xlim',limx,'ylim',[0 1])
        else
            xlabel('WAKE: Time relative to image onset (s)');
            yl=ylabel('\color{black}SLEEP: Time relative to sound onset (s)');
            axes('position',[0.13 0.1244 0.1566 0.02],'xtick',[],'ytick',[]);hold all;fill([0 0 1 1],[0 1 1 0],[1 1 1]*0.8,'edgecolor','none');text(0.5,0.5,'Abstract','color','k','horizontalalignment','center','verticalalignment','middle');axis off;set(gca,'xlim',limx,'ylim',[0 1])
        end
    end
    set(yl,'position',[-0.3+limx(1)    3.7500    1.0000])

    subplot(12,4,2:4:18);

    imagesc(tm,tm,nanmean(smootheddataRSA(:,:,:,3),3)');
    set(gca,'ydir','normal','xlim',limx,'ylim',limy,'clim',[-0.05 0.05]);
    clrmap=colormap(gca);
    set(gca,'fontsize',12,'ydir','normal','ticklength',[0.03 0],'xtick',[-1:0.5:2.5],'xticklabel',{'\color{black}-1','\color{black}-0.5','','','','\color{black}1.5','\color{black}2','\color{black}2.5'},'ytick',-1:0.5:2.5,'yticklabel',{'-1','-0.5',char([55357,56586]),'0.5','1','1.5','2','2.5'});
    ylabel('\color{black}SLEEP: Time relative to sound onset (s)');
    xl=xlabel('WAKE: Time relative to image onset (s)');
    hold all
    if limy_for_analysis(1)~=limy(1) && limy_for_analysis(2)~=limy(2)
        fill([limx(1)+0.05*diff(limx),limx(1)+0.05*diff(limx),limx(2)-0.05*diff(limx),limx(2)-0.05*diff(limx)],[limy_for_analysis(1),limy_for_analysis(2),limy_for_analysis(2),limy_for_analysis(1)],clrmap(jj,:),'facecolor','none','edgecolor',[1 1 1]*0.9,'linestyle',':');
    end
    line([0 0],limy,'color','k');
    line(limx,[0 0],'color','k');
    hold all;
    fill([cbar(1,1) cbar(1,2) cbar(1,2) cbar(1,1)],[cbar(2,1) cbar(2,1) cbar(2,2) cbar(2,2)],'w');
    for jj=1:size(clrmap,1)
        fill(cbar(1,1)+diff(cbar(1,:))*([0 1 1 0]+jj-1)/size(clrmap,1),[cbar(2,1) cbar(2,1) cbar(2,2) cbar(2,2)],clrmap(jj,:),'edgecolor','none');
    end
    text(cbar(1,1)+0.02,mean(cbar(2,:),2),'-0.05','horizontalalignment','left','verticalalignment','middle','fontsize',10,'color','w');
    text(mean(cbar(1,:)),mean(cbar(2,:),2),'\Delta{\itr}','horizontalalignment','center','verticalalignment','middle','fontsize',12,'color','w');
    text(cbar(1,2)-0.02,mean(cbar(2,:),2),'0.05','horizontalalignment','right','verticalalignment','middle','fontsize',10);
    axes('position',[0.3361 0.5568 0.1566 0.04],'xtick',[],'ytick',[]);hold all;fill([0 0 1 1],[0 1 1 0],'k','edgecolor','none');text(0.5,0.5,'Place','color',[1 1 1]*0.999,'horizontalalignment','center','verticalalignment','middle');fill([0 0 1 1],1+[0 1 1 0],[1 1 1]*0.8,'edgecolor','none');text(0.5,1.5,'Abstract','color','k','horizontalalignment','center','verticalalignment','middle');axis off;set(gca,'xlim',limx,'ylim',[0 2])
    set(xl,'position',[mean(limx) -2.05 1]);
    
    subplot(48,4,98:4:180);
    hold all;

    dat=squeeze(nanmean(smootheddataRSA(:,tm>=limy_for_analysis(1) & tm<=limy_for_analysis(2),:,3),2));
    clusters=[0.3 0.4;0.45 0.7;0.9 1.2];
    for ii=1:size(clusters,1)
        indices=find(tm>=clusters(ii,1) & tm<=clusters(ii,2));
        [peak_val,peak_idx]=max(abs(nanmean(dat(indices,:),2)));
        clusters(ii,1)=find(abs(nanmean(dat(1:indices(peak_idx),:),2))<=peak_val/2,1,'last');clusters(ii,1)=tm(clusters(ii,1)+1);
        clusters(ii,2)=indices(peak_idx)-1+find(abs(nanmean(dat(indices(peak_idx):end,:),2))<=peak_val/2,1);clusters(ii,2)=tm(clusters(ii,2)-1);
        clusters(ii,:)=[tm(find(tm>=clusters(ii,1),1)),tm(find(tm<=clusters(ii,2),1,'last'))];
        fill([clusters(ii,1) clusters(ii,1) clusters(ii,2) clusters(ii,2)],[-1 1 1 -1],[1 1 1],'FaceColor',clrs(ii,:),'EdgeColor','none','LineStyle','-','linewidth',2,'FaceAlpha',0.1)
        res(:,ii)=nanmean(dat(tm>=clusters(ii,1) & tm<=clusters(ii,2),:),1)';
    end
    set(gca,'xlim',limx,'fontsize',12,'xtick',[-0.5 1.5 2],'xticklabel',{'\color{black}-0.5','\color{black}1.5','\color{black}2'});
    xl=xlabel('WAKE: Time relative to image onset (s)');
    ylabel('Scene specific reactivation (\Delta{\itr})');
    boundedline(tm,nanmean(dat,2),nanstd(dat,[],2)/sqrt(size(smootheddataRSA,3)),'cmap',cl(42,:),'alpha','transparency',0.35)
    plot(tm,nanmean(dat,2),'linewidth',2);
    line([tm(1),tm(length(tm))],[0 0]); %chance line
%     limit=[min(nanmean(nanmean(smootheddataRSA(:,tm>=limy_for_analysis(1) & tm<=limy_for_analysis(2),:,3),3),2)-nanstd(nanmean(smootheddataRSA(:,tm>=limy_for_analysis(1) & tm<=limy_for_analysis(2),:,3),2),[],3)/sqrt(size(smootheddataRSA,3))),max(nanmean(nanmean(smootheddataRSA(:,tm>=limy_for_analysis(1) & tm<=limy_for_analysis(2),:,3),3),2)+nanstd(nanmean(smootheddataRSA(:,tm>=limy_for_analysis(1) & tm<=limy_for_analysis(2),:,3),2),[],3)/sqrt(size(smootheddataRSA,3)))];
    limit=[-0.04 0.04];
    ylim(limit);
    line([0 0],[-1 1]);
    set(xl,'position',[mean(limx)   -0.0505   -1.0000]);
    drawnow;
    set(gca,'ytick',sort([0 max(get(gca,'ytick')) min(get(gca,'ytick'))]));

    axes('position',[0.3361 0.1212 0.1566 0.04],'xtick',[],'ytick',[]);hold all;
    fill([0 0 1 1],[0 1 1 0],'k','edgecolor','none');
    text(0.5,0.5,'Place','color',[1 1 1]*0.999,'horizontalalignment','center','verticalalignment','middle');
    fill([0 0 1 1],1+[0 1 1 0],[1 1 1]*0.8,'edgecolor','none');
    text(0.5,1.5,'Abstract','color','k','horizontalalignment','center','verticalalignment','middle');axis off;
    set(gca,'xlim',limx,'ylim',[0 2])

    
    behav=load('SubdataAllSubs.mat');
    % First column is the effect on the non-cued items in a cued scene, second is the effect on cued items 
    [~,subloc1,subloc2]=intersect(behav.subnums,subnums);
    
    for clusternum=1:size(clusters,1)
        [r(clusternum,1),p_r(clusternum,1)]=corr(behav.subdata(subloc1,1),res(subloc2,clusternum));
        disp(['Correlation between within-sound power-ICC (cluster #',num2str(clusternum),') and non-cued benefit is r=',num2str(r(clusternum,1)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,1)),' (2way)']);
        [r(clusternum,2),p_r(clusternum,2)]=corr(behav.subdata(subloc1,2),res(subloc2,clusternum));
        disp(['Correlation between within-sound power-ICC (cluster #',num2str(clusternum),') and cued benefit is r=',num2str(r(clusternum,2)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,2)),' (2way)']);
    end
    
    figure;
    set(gcf,'windowstyle','normal','windowstate','maximized');

    for clusternum=1:size(clusters,1)
        for corrtype=1:2
            subplot('position',[0.0638+clusternum*0.113 1.009-0.2697*corrtype 0.09    0.2157]);
            hold all
            x=res(subloc2,clusternum); 
            y=behav.subdata(subloc1,corrtype); 
            [x,k] = sort(x);
            y = y(k);
            surface_area_model = fitlm(x,y,'linear');
            [p,s]=polyfit(x,y,1);
            [yfit,dy]=polyconf(p,x,s,'predopt','curve');
            a= 25;
            h1= scatter(x,y,a,'o','LineWidth',0.5,'MarkerEdgeColor',clrs(clusternum,:),'MarkerFaceColor',clrs(clusternum,:));
            line(x,yfit,'color',clrs(clusternum,:),'LineWidth',0.5);
            line(x,yfit-dy,'color',clrs(clusternum,:),'linestyle',':');
            line(x,yfit+dy,'color',clrs(clusternum,:),'linestyle',':');
            drawnow;
            switch clusternum
                case 1
                    xlim([-0.2 0.1]);
                case 2
                    xlim([-0.1 0.15]);
                case 3
                    xlim([-0.15 0.1]);
            end
            switch corrtype
                case 1
                    ylim([-1.5 1.5]);
                case 2
                    ylim([-1.5 1.5]);
                    if clusternum==ceil(size(clusters,1)/2)
                        xlabel('Scene specific reactivation (\Delta{\itr})');
                    end
            end
            if clusternum==1
                if corrtype==1
                    ylab=ylabel('Uncued items');
                else
                    ylab=ylabel('Cued items');
                end
            else
                set(gca,'yticklabels',{});
            end
            if corrtype==1
                set(gca,'xticklabels',{});
            end

            
            drawnow;
            set(gca,'xtick',sort([0,get(gca,'xlim')]),'ytick',sort([0,get(gca,'ylim')]),'fontsize',12);
            line([0 0],[min(get(gca,'ylim'))+0.07*(diff(get(gca,'ylim'))) max(get(gca,'ylim'))],'color','k');
            line(get(gca,'xlim'),[0 0],'color','k');
            text(mean(get(gca,'xlim')),min(get(gca,'ylim')),['r = ',num2str(round(100*r(clusternum,corrtype))/100),', {\itp} < ',num2str(ceil(100*p_r(clusternum,corrtype))/100),' '],'color',clrs(clusternum,:),'horizontalalignment','center','verticalalignment','bottom');
        end
    end
    axes('position',[0.185 0.4696 0.9 0.4854],'fontsize',12);axis off
    a=text(0.5,0.5,'\color{black}\Delta memory - items in cued scenes (Z-score)\newline                better <---------> worse','rotation',90,'fontsize',16);
    set(a,'position',[-0.0720 0.5 0],'horizontalalignment','center');

catch err
    show_err(err,0);
    1
end