
try
    clear;
    DataFold=[pwd,'\EEG\'];
    numperms=10000;
    elecname='Cz';
    Thresh=0.01;
    load([pwd,'\Datasets\sublists']);
    subnums=subs_atleast3itemrecall;
    m_file_text=fileread(matlab.desktop.editor.getActiveFilename);
    m_file_name=matlab.desktop.editor.getActiveFilename;
    datamat=[];
    ERP=[];
    for sub = 1:length(subnums)       
        clearvars stageData data_trial;
        disp(['Subject #',num2str(sub),'/',num2str(length(subnums))]);
        load([DataFold,num2str(subnums(sub)),'s.mat'],'data_trial')
        load([DataFold,'United\',num2str(subnums(sub)),'sScore.mat']);
        elec=find(cellfun(@(x)sum(strcmp(x,elecname)),data_trial.label));
        stageData.onsets(end)=stageData.onsets(end-1)+(stageData.onsets(2)-stageData.onsets(1));
        stage_per_trial=stageData.stages(max(bsxfun(@gt,[data_trial.cfg.trl(:,1)-data_trial.cfg.trl(:,3)]',stageData.onsets).*repmat((1:length(stageData.onsets)),[size(data_trial.cfg.trl,1),1])',[],1)');
        rel_trials=find(stage_per_trial==2 | stage_per_trial==3);
        for ii=1:length(rel_trials)
            [tmp,f,t]=spectrogram(nanmean(data_trial.trial{rel_trials(ii)}(elec,:),1)'-mean(nanmean(data_trial.trial{rel_trials(ii)}(elec,:),1)),256,224,0.25:0.25:25,data_trial.fsample);
            t=t+data_trial.time{1}(1);
            if ii==1 && sub==1
                specmat=nan(size(tmp,1),size(tmp,2),length(subnums),1000);
            end
            specmat(:,:,sub,ii)=abs(tmp);
            ERP(end+1,:)=nanmean(data_trial.trial{rel_trials(ii)}(elec,:),1)';
            datamat(end+1,:)=[subnums(sub),data_trial.trialinfo(ii)];
        end
    end
    datamat(:,3)=datamat(:,2)~=76;
    
    specmat=specmat(:,:,:,1:find(~isnan(squeeze(nanmean(nanmean(nanmean(specmat,3),2),1))),1,'last'));
    specmat_z=(specmat-repmat(nanmean(specmat(:,t<0,:,:),2),[1,size(specmat,2),1,1]))./repmat(nanstd(specmat(:,t<0,:,:),[],2),[1,size(specmat,2),1,1]);
    specmat_z_persub=(specmat-repmat(nanmean(nanmean(specmat(:,t<0,:,:),2),4),[1,size(specmat,2),1,size(specmat,4)]))./repmat(nanmean(nanstd(specmat(:,t<0,:,:),[],2),4),[1,size(specmat,2),1,size(specmat,4)]);
    specmat_perc=100*(specmat-repmat(nanmean(specmat(:,t<0,:,:),2),[1,size(specmat,2),1,1]))./repmat(nanmean(specmat(:,t<0,:,:),2),[1,size(specmat,2),1,1]);
    specmat_perc_persub=100*(specmat-repmat(nanmean(nanmean(specmat(:,t<0,:,:),4),2),[1,size(specmat,2),1,size(specmat,4)]))./repmat(nanmean(nanmean(specmat(:,t<0,:,:),4),2),[1,size(specmat,2),1,size(specmat,4)]);
    
    dataset=specmat_perc_persub;
    
    ERP_per_sub=nan(length(subnums),size(ERP,2));
    for sub=1:length(subnums)
        ERP_per_sub(sub,:)=nanmean(ERP(datamat(:,1)==subnums(sub),:),1);
    end
    
%% more simple stats, based on unweighed between sub ttests per data point
    statmat_across_ppt=nan(size(dataset,1),size(dataset,2),2);
    for ii=1:size(dataset,1)
        for jj=1:size(dataset,2)
            tmp=squeeze(nanmean(dataset(ii,jj,:,:),4))';
            statmat_across_ppt(ii,jj,1)=mean(tmp);
            [~,statmat_across_ppt(ii,jj,2)]=ttest(tmp);
        end
    end
    dataset_only76=dataset;
    dataset_no76=dataset;
    for sub=1:length(subnums)
        idx=datamat(datamat(:,1)==subnums(sub),2);
        dataset_only76(:,:,sub,idx~=76)=nan;
        dataset_no76(:,:,sub,idx==76)=nan;
    end
        
    %% find two biggest clusters based on the between subjects analysis
    Pthreshold=statmat_across_ppt(:,:,2)<Thresh/(size(statmat_across_ppt,1)*size(statmat_across_ppt,2));
    all_clust=bwconncomp(Pthreshold);% Finds 2d cluster sizes
    numPixels = cellfun(@numel,all_clust.PixelIdxList);[~,numPixels]=sort(numPixels,'descend');
    cluster{1}=all_clust.PixelIdxList{numPixels(1)};
    cluster{2}=all_clust.PixelIdxList{numPixels(2)};
    threshold_in_time=1;%s
    cluster{3}=cluster{2}(cluster{2}<=(find(t>=threshold_in_time,1)-1)*length(f)); %  made to exclude only spindle-classic activity, <1s
    if min(mod(cluster{2},length(f)))<min(mod(cluster{1},length(f)))
        cluster{4}=cluster{1}(cluster{1}>(find(t>=threshold_in_time,1)-1)*length(f)); %  made to include only spindle-classic activity, >=1s
    else
        cluster{4}=cluster{2}(cluster{2}>(find(t>=threshold_in_time,1)-1)*length(f)); %  made to include only spindle-classic activity, >=1s
    end
    clust_names={'Delta-theta','All sigma','Early sigma (<=1s)','Late sigma (>1s)'};
    if length(cluster)>3 && isempty(cluster{4})
        cluster=cluster(1:3);
        clust_names=clust_names(1:3);
    end
    for sub=1:length(subnums)
        idx=find(datamat(:,1)==subnums(sub));
        for jj=1:length(idx)
            tmp=dataset(:,:,sub,jj);
            for clusternum=1:length(cluster)
                datamat(idx(jj),clusternum+3)=nanmean(tmp(cluster{clusternum}));
            end
        end
    end
    Pthreshold2=zeros(size(Pthreshold));
    for clusternum=1:length(cluster)
        Pthreshold2(cluster{clusternum})=clusternum;
    end

    
    %% ICC analysis
    min_num_of_presentations=5;
    real_subnum1=[];
    for clusternum=1:length(cluster)
        ICCmat{clusternum}=[];
        p_vals=[];
        datamattmp=datamat;
        for sub=1:length(subnums)
            snds=unique(datamat(datamat(:,1)==subnums(sub),2));
            mat=nan(length(snds),100);
            for ii=1:length(snds)
                tmp=find(datamat(:,1)==subnums(sub) & datamat(:,2)==snds(ii));
                mat(ii,1:length(tmp))=datamat(tmp(randperm(length(tmp))),clusternum+3)';
            end
            if find(~isnan(sum(mat,1)),1,'last')<min_num_of_presentations
                continue;
            elseif clusternum==1
                real_subnum1=[real_subnum1;[subnums(sub),find(~isnan(sum(mat,1)),1,'last')]];
            end
            if icc_r
                mat=mat(:,1:find(~isnan(nansum(mat,1)),1,'last'));
                ICCmat{clusternum}(end+1,:)=ICC_rmany_w_rand(mat,numperms);                
            else
                [ICCmat{clusternum}(end+1,1),~,~,~,~,~,p_vals(end+1,1)]=ICC(mat,'1-k');
                idx=find(datamat(:,1)==subnums(sub));
                for perm=1:numperms
                    datamattmp(idx,clusternum+3)=datamat(idx(randperm(length(idx))),clusternum+3);
                    mat=nan(length(snds),100);
                    for ii=1:length(snds)
                        tmp=find(datamattmp(:,1)==subnums(sub) & datamat(:,2)==snds(ii));
                        mat(ii,1:length(tmp))=datamattmp(tmp(randperm(length(tmp))),clusternum+3)';
                    end
                    [ICCmat{clusternum}(end,perm+1),~,~,~,~,~,p_vals(end,perm+1)]=ICC(mat,'1-k');            
                end
            end
        end    
    end

   
    %% ICC analysis for items belonging to the same scene
    min_num_of_presentations=5;
    real_subnum2=[];
    clearvars ic;
    for clusternum=1:length(cluster)
        ICCmat2{clusternum}=[];ICCmat3{clusternum}=[];
        p_vals2=[];p_vals3=[];
        datamattmp=datamat;
        for ii=1:numperms+1
            mat4{ii}=nan(0,2);
        end
        for sub=1:length(subnums)
            snds=unique(datamat(datamat(:,1)==subnums(sub),2)); % 12 sounds altogether + 1 novel sound
            if length(snds)<13 % in case not all sounds were presented at least once
                snds=[0;snds];%continue;
            end
            mat=nan(length(snds),100);
            for ii=1:length(snds)
                tmp=find(datamat(:,1)==subnums(sub) & datamat(:,2)==snds(ii)); % all instances in which a specific sound was presented for a specific participant
                mat(ii,1:length(tmp))=datamat(tmp(randperm(length(tmp))),clusternum+3)'; % insert into the columns all presentations of the sound; the remaining solumns have NaNs 
            end
            mat=sort(mat,2);
            if isempty(find(~isnan(sum(mat,1)),1,'last')) || find(~isnan(sum(mat,1)),1,'last')<min_num_of_presentations
                continue;
            elseif clusternum==1
                real_subnum2=[real_subnum2;[subnums(sub),find(~isnan(sum(mat,1)),1,'last')]];
            end
            idxs=[1 3 5 7 9 11 2 4 6 8 10 12]; % The first six are six sounds, the other are the respective other six sounds from the same scene (i.e., 1<-->2, 3<--4> etc)
            mat2=[mat(idxs(1:6),:),mat(idxs(7:12),:)];mat2=sort(mat2,2); % All presentations of all scene related scenes are in the same line, regardless of sounds identity
            mat3=[nanmean(mat(idxs(1:6),:),2),nanmean(mat(idxs(7:12),:),2)];mat3=sort(mat3,2); %Here, the average activity for each of a pair of sounds within a scene is calculated, so there are only two columns/numbers for each scene, one for each sound (averaged)
            mat3real=mat3;
            mat4{1}(end+1:end+6,:)=mat3;% mat4 takes all mat3 pairs together for all subjects (not very useful, because the between subject component of variance is not modeled)
            [ICCmat2{clusternum}(end+1,1),~,~,~,~,~,p_vals2(end+1,1)]=ICC(mat2,'1-k');
            [ICCmat3{clusternum}(end+1,1),~,~,~,~,~,p_vals3(end+1,1)]=ICC(mat3,'1-k');
            for perm=1:numperms
                idxs=idxs(randperm(length(idxs))); % sounds are still clustered together with other presenations of the same sound, but the pairing is shuffled
                mat2=[mat(idxs(1:6),:),mat(idxs(7:12),:)];mat2=sort(mat2,2);
                mat3=[nanmean(mat(idxs(1:6),:),2),nanmean(mat(idxs(7:12),:),2)];mat3=sort(mat3,2);
                mat4{1+perm}(end+1:end+6,:)=mat3;
                mat5=reshape(mat3real(randperm(numel(mat3real))),size(mat3real));
                [ICCmat2{clusternum}(end,perm+1),~,~,~,~,~,p_vals2(end,perm+1)]=ICC(mat2,'1-k');            
                [ICCmat3{clusternum}(end,perm+1),~,~,~,~,~,p_vals3(end,perm+1)]=ICC(mat3,'1-k');            
            end
        end    
        figure;hist(nanmean(ICCmat3{clusternum}(:,2:end),1));
        drawnow;
        line([1 1]*nanmean(ICCmat3{clusternum}(:,1),1),[0 max(get(gca,'ylim'))]);
        p=length(find((ICCmat3{clusternum}(:,2:end)-repmat(nanmean(ICCmat3{clusternum}(:,2:end),2),[1,numperms]))>(repmat(ICCmat3{clusternum}(:,1),[1,numperms])-repmat(nanmean(ICCmat3{clusternum}(:,2:end),2),[1,numperms]))))/(numperms*size(ICCmat3{clusternum},1));
        [~,p2(clusternum,2)]=ttest(nanmean(ICCmat3{clusternum}(:,2:end),2),ICCmat3{clusternum}(:,1),'tail','left');
        title(['ICC within-scene: AVERAGED random perm results for cluster = ',num2str(clusternum),' (',clust_names{clusternum},'), p_a_l_l_p_e_r_m_s = ',num2str(p),', p_t_t_e_s_t _v_s _m_e_a_n = ',num2str(p2(clusternum,2)),', numperms = ',num2str(numperms),', min_num_of_presentations = ',num2str(min_num_of_presentations),', line signifies real data']);
    
    end
    %% Demonstrration that ICC is negatively biased for random values:
    clearvars nnn nn;
    values=round(exp(1.4:0.2:7));
    for kk=1:length(values)
        for ii=1:1e4
            nn(ii)=ICC(rand(values(kk),10),'1-k');
        end;
        nnn(kk)=mean(nn);
    end;
    figure;
    plot(values,nnn);
    set(gca,'xscale','log','yscale','linear')
    title('Negative bias for ICC (1-k) as a function of number of classes');
    xlabel('Number of classes (log scale)');
    ylabel('       ICC calculated for random values\newline(10 values per class; 10,000 permutations)');
    drawnow;
    line([6 6],get(gca,'ylim'),'linestyle',':');line([12 12],get(gca,'ylim'),'linestyle','--');
    %% Correlations with behavior
    behav=load('Subdata.mat');
    % First column is the effect on the non-cued items in a cued scene, second is the effect on cued items 
    [~,subloc1,subloc2]=intersect(behav.subnums,real_subnum1(:,1));
    for clusternum=1:length(cluster)
        ZIcc{clusternum,1}=(ICCmat{clusternum}(:,1)-nanmean(ICCmat{clusternum},2))./nanstd(ICCmat{clusternum},[],2);
        [r(clusternum,1),p_r(clusternum,1)]=corr(behav.subdata(subloc1,1),ZIcc{clusternum,1}(subloc2));
        disp(['Correlation between within-sound power-ICC (',clust_names{clusternum},') and non-cued benefit is r=',num2str(r(clusternum,1)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,1)),' (2way)']);
        [r(clusternum,2),p_r(clusternum,2)]=corr(behav.subdata(subloc1,2),ZIcc{clusternum,1}(subloc2));
        disp(['Correlation between within-sound power-ICC (',clust_names{clusternum},') and cued benefit is r=',num2str(r(clusternum,2)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,2)),' (2way)']);
    end
    
    behav.subdata(:,6)=squeeze(behav.tab(:,2,1)-behav.tab(:,1,1));
    behav.subdata(:,7)=squeeze(behav.tab(:,2,2)-behav.tab(:,1,2));
    behav.subdata(:,8)=squeeze((behav.tab(:,2,1)-behav.tab(:,1,1))./(behav.tab(:,3,1)-behav.tab(:,1,1)));
    behav.subdata(:,9)=squeeze((behav.tab(:,2,2)-behav.tab(:,1,2))./(behav.tab(:,3,2)-behav.tab(:,1,2)));
    reg_or_zscore=1;
    [~,subloc1,subloc2]=intersect(behav.subnums,real_subnum2(:,1));
    for clusternum=1:length(cluster)
        if reg_or_zscore==2
            ZIcc{clusternum,2}=(ICCmat3{clusternum}(:,1)-nanmean(ICCmat3{clusternum},2))./nanstd(ICCmat3{clusternum},[],2);
        elseif reg_or_zscore==1
            ZIcc{clusternum,2}=ICCmat3{clusternum}(:,1);%-nanmean(ICCmat3{clusternum},2))./nanstd(ICCmat3{clusternum},[],2);
        end
        behav.subdata(:,6)=squeeze(behav.tab(:,2,1)-behav.tab(:,1,1));behav.subdata(:,7)=squeeze(behav.tab(:,2,2)-behav.tab(:,1,2));behav.subdata(:,8)=squeeze((behav.tab(:,2,1)-behav.tab(:,1,1))./(behav.tab(:,3,1)-behav.tab(:,1,1)));behav.subdata(:,9)=squeeze((behav.tab(:,2,2)-behav.tab(:,1,2))./(behav.tab(:,3,2)-behav.tab(:,1,2)));        
        [r(clusternum,3),p_r(clusternum,3)]=corr(behav.subdata(subloc1,1),ZIcc{clusternum,2}(subloc2));
        disp(['Correlation between within-scene power-ICC (',clust_names{clusternum},') and non-cued benefit is r=',num2str(r(clusternum,3)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,3)),' (2way)']);
        [r(clusternum,4),p_r(clusternum,4)]=corr(behav.subdata(subloc1,2),ZIcc{clusternum,2}(subloc2));
        disp(['Correlation between within-scene power-ICC (',clust_names{clusternum},') and cued benefit is r=',num2str(r(clusternum,4)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,4)),' (2way)']);
        [r(clusternum,5),p_r(clusternum,5)]=corr(behav.subdata(subloc1,3),ZIcc{clusternum,2}(subloc2));
        disp(['Correlation between within-scene power-ICC (',clust_names{clusternum},') and non-cued (in non-cued scene) error is r=',num2str(r(clusternum,5)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,5)),' (2way)']);
        [r(clusternum,6),p_r(clusternum,6)]=corr(behav.subdata(subloc1,4),ZIcc{clusternum,2}(subloc2));
        disp(['Correlation between within-scene power-ICC (',clust_names{clusternum},') and basic forgetting slope for ppt is r=',num2str(r(clusternum,6)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,6)),' (2way)']);
        [r(clusternum,7),p_r(clusternum,7)]=corr(behav.subdata(subloc1,5),ZIcc{clusternum,2}(subloc2));
        disp(['Correlation between within-scene power-ICC (',clust_names{clusternum},') and non-cued (in non-cued scene) "benefit" is r=',num2str(r(clusternum,7)),' (negative = more benefit with power), p=',num2str(p_r(clusternum,7)),' (2way)']);
    end

%% General figure for paper/poster
    drawoption=1;
    pval_or_stars=2;
    colormap default
    clrs=[0.3 0.3 0.3;0.7 0.7 0.7;0 1 0;0 0 1];
    climits=[-10 130];
    figure(1); 
    set(gcf,'windowstyle','normal','windowstate','maximized');
    clf
    subplot(2,12,1:4);drawnow;
    set(gca,'position',get(gca,'position')-[0.025 0 0 0]);
    imagesc(t,f,statmat_across_ppt(:,:,1));set(gca,'ydir','normal')
    set(gca,'clim',climits)
    colormap(gca,'jet');
    caxis(climits);
    clrmap=colormap;
    set(gca,'fontsize',12,'ydir','normal','ticklength',[0.03 0],'xlim',[min(t) max(t)],'ylim',[min(f) max(f)],'xticklabel',{'-1',char([55357,56586]),'1','2','3','4'});
    line([0 0],[f(1) f(end)],'color','k');
    ylabel('\color{black}Hz');
    xlabel('\color{black}Time relative to sound onset (s)');
    
    hold all;
    fill([2 4 4 2],[21 21 24 24],'w');
    for jj=1:size(clrmap,1)
        fill(2+2*([0 1 1 0]+jj-1)/size(clrmap,1),[21.1 21.1 23.9 23.9],clrmap(jj,:),'edgecolor','none');
    end
    text(2,22.5,[num2str(climits(1)),'%'],'horizontalalignment','left','verticalalignment','middle','fontsize',10,'color','w');
    text(4,22.5,[num2str(climits(2)),'%'],'horizontalalignment','right','verticalalignment','middle','fontsize',10);
    
    axes('position',get(gca,'position'));
    plot(data_trial.time{1},nanmean(ERP_per_sub,1),'color',[1 0 1],'linewidth',2);
    set(gca,'fontsize',12,'color','none','yaxislocation','right','xtick',[],'ytick',[-25 0 15],'ycolor',[1 0 1],'ylim',[-25 15],'xlim',[min(t) max(t)],'xticklabel',{'-1',char([55357,56586]),'1','2','3','4'});
    ylabel('\muV','rotation',0)
    
    
    subplot(2,12,5:8);
    imagesc(t,f,statmat_across_ppt(:,:,2)*size(statmat_across_ppt,1)*size(statmat_across_ppt,2));
    set(gca,'clim',[0 Thresh],'ydir','normal');
    colormap(gca,'bone');
    caxis([0 Thresh]);
    xlabel('\color{black}Time relative to sound onset (s)');
    clrmap=colormap(gca);
    set(gca,'fontsize',12,'ydir','normal','ticklength',[0.03 0],'xlim',[min(t) max(t)],'ylim',[min(f) max(f)],'yticklabels',{},'xticklabel',{'-1',char([55357,56586]),'1','2','3','4'});
    line([0 0],[f(1) f(end)],'color','k');
    hold all;
    fill([2 4 4 2],[21 21 24 24],'w');
    for jj=1:size(clrmap,1)
        fill(2+2*([0 1 1 0]+jj-1)/size(clrmap,1),[21.1 21.1 23.9 23.9],clrmap(jj,:),'edgecolor','none');
    end
    text(2.1,22.5,'{\itp}=0','horizontalalignment','left','verticalalignment','middle','fontsize',10,'color','w');
    text(4,22.5,['{\itp}>',num2str(Thresh)],'horizontalalignment','right','verticalalignment','middle','fontsize',10);
    
    
    subplot(2,12,9:12);drawnow;
    set(gca,'position',get(gca,'position')+[0.025 0 0 0]);
    Pthreshold3=Pthreshold2;
    Pthreshold3(Pthreshold2>=2)=2;
    imagesc(t,f,Pthreshold3);
    try
        line([0 0],[f(1) f(end)],'color','w');
        set(gca,'ydir','normal','clim',[0 2]);
        set(gca,'xcolor',[1 1 1],'ycolor',[1 1 1]','fontsize',12,'ydir','normal','ticklength',[0.00 0],'xlim',[min(t) max(t)],'ylim',[min(f) max(f)],'yticklabels',{});
        xlabel('\color{black}Time relative to sound onset (s)');
        xticklabels({'\color{black}-1',['\color{black}',char([55357,56586])],'\color{black}1','\color{black}2','\color{black}3','\color{black}4'});
        colormap(gca,[0 0 0;0 0 0;clrs(1,:);clrs(2,:)])
        hold all;
        tryclust=bwboundaries(Pthreshold2(:,1:(find(t>=threshold_in_time,1)-1)));
        fill(-0.01+t(tryclust{2}(:,2)),f(tryclust{2}(:,1)),[1 1 1],'FaceColor','none','EdgeColor',clrs(3,:),'LineStyle',':','linewidth',2)
        text(t(round(mean(tryclust{1}(:,2)))),f(round(mean(tryclust{1}(:,1)))),'1','color','w','horizontalalignment','center','FontWeight','Bold','fontsize',12);
        text(t(round(mean(tryclust{2}(:,2)))),18.62,'2A','color',clrs(3,:),'horizontalalignment','center','FontWeight','Bold','fontsize',12);
        text(0.8,14,'2','color',clrs(2,:),'horizontalalignment','center','FontWeight','Bold','fontsize',12);     
        tryclust=bwboundaries(Pthreshold2(:,(find(t>=threshold_in_time,1)-1):end));
        fill(0.01+t((find(t>=threshold_in_time,1)-1)-1+tryclust{1}(:,2)),f(tryclust{1}(:,1)),[1 1 1],'FaceColor','none','EdgeColor',clrs(4,:),'LineStyle',':','linewidth',2)
        text(t(round(mean((find(t>=threshold_in_time,1)-1)-1+tryclust{1}(:,2)))),f(round(mean(tryclust{1}(:,1)))),'2B','color',clrs(4,:),'horizontalalignment','center','FontWeight','Bold','fontsize',12);
    catch,end
    
    if drawoption==1
        subplot(2,12,13:15);
    else
        subplot('position',[0.1244    0.1100    0.1159    0.3412]);
    end
    hold all;
    for clusternum=1:length(cluster)
        line([1,2]+3*(clusternum-1),[ICCmat{clusternum}(:,1),nanmean(ICCmat{clusternum}(:,2:end),2)],'color',[1 1 1]*0.9,'Marker','o','markersize',2);
        bar(1:11,[nan(1,(clusternum-1)*3),nanmean(ICCmat{clusternum}(:,1)),nanmean(nanmean(ICCmat{clusternum}(:,2:end))),nan(1,9-(clusternum-1)*3)],'facecolor',clrs(clusternum,:),'edgecolor',[0 0 0]);
        line(((clusternum-1)*3+1)*[1 1],nanmean(ICCmat{clusternum}(:,1))+((-1)^((nanmean(ICCmat{clusternum}(:,1))>0)+1))*[0 nanstd(ICCmat{clusternum}(:,1))/sqrt(size(ICCmat{clusternum},1))],'color',[0 0 0]);
        line(((clusternum-1)*3+2)*[1 1],nanmean(nanmean(ICCmat{clusternum}(:,2:end)))+((-1)^((nanmean(nanmean(ICCmat{clusternum}(:,2:end)))>0)+1))*[0 nanstd(nanmean(ICCmat{clusternum}(:,2:end),2),[],1)/sqrt(size(ICCmat{clusternum},1))],'color',[0 0 0]);
        line(((clusternum-1)*3+[1 2]),0.5*[1 1],'color',[0 0 0]);
        shift=0;
        if p2(clusternum,1)<0.001
            if pval_or_stars==1,ptext='{\itp}<0.001';else,ptext='***';shift=0.025;end
        elseif p2(clusternum,1)<0.01
            if pval_or_stars==1,ptext='{\itp}<0.01';else,ptext='**';shift=0.025;end
        elseif p2(clusternum,1)<0.05
            if pval_or_stars==1,ptext='{\itp}<0.05';else,ptext='*';shift=0.025;end
        elseif p2(clusternum,1)<0.1
            if pval_or_stars==1,ptext='{\itp}<0.1';else,ptext='~';end
        else
            if pval_or_stars==1,ptext=['{\itp}=',num2str(round(100*p2(clusternum,1))/100)];else,ptext='n.s';end
        end
        text(((clusternum-1)*3+1.5),0.5-shift,ptext,'horizontalalignment','center','verticalalignment','bottom','fontsize',16);
    end
    set(gca,'xtick',[1,2,4,5,7,8,10,11],'xticklabel',{'Real','Shuf','Real','Shuf','Real','Shuf','Real','Shuf','Real','Shuf'},'fontsize',16);
    xtickangle(gca,90);
    ylabel('ICC within-object');
    drawnow;
    if drawoption==1
        set(gca,'position',get(gca,'position').*[1 1 0.8 1]);
    end
    
    for ybreak=1:2
        if drawoption==1
            if ybreak==1
                subplot(2,12,16:18);drawnow;
                tmppos=get(gca,'position');
                set(gca,'position',get(gca,'position').*[1 1 1 0.1]);
            else
                tmppos(2)=tmppos(2)+tmppos(4)*0.12;
                tmppos(4)=tmppos(4)*0.88;
                subplot('Position',tmppos);
            end
        else
            subplot('position',[0.2845    0.1100    0.1159    0.3412]);
        end
        
        hold all;
        for clusternum=1:length(cluster)
            data_for_scatter_plot=ICCmat3{clusternum};
            if ybreak==2 && clusternum==3 
                data_for_scatter_plot(7,1)=-3.2;
            end
            line([1,2]+3*(clusternum-1),[data_for_scatter_plot(:,1),nanmean(data_for_scatter_plot(:,2:end),2)],'color',[1 1 1]*0.9,'Marker','o','markersize',2);
            bar(1:11,[nan(1,(clusternum-1)*3),nanmean(ICCmat3{clusternum}(:,1)),nanmean(nanmean(ICCmat3{clusternum}(:,2:end))),nan(1,9-(clusternum-1)*3)],'facecolor',clrs(clusternum,:),'edgecolor',[0 0 0]);
            line(((clusternum-1)*3+1)*[1 1],nanmean(ICCmat3{clusternum}(:,1))+((-1)^((nanmean(ICCmat3{clusternum}(:,1))>0)+1))*[0 nanstd(ICCmat3{clusternum}(:,1))/sqrt(size(ICCmat3{clusternum},1))],'color',[0 0 0]);
            line(((clusternum-1)*3+2)*[1 1],nanmean(nanmean(ICCmat3{clusternum}(:,2:end)))+((-1)^((nanmean(nanmean(ICCmat3{clusternum}(:,2:end)))>0)+1))*[0 nanstd(nanmean(ICCmat3{clusternum}(:,2:end),2),[],1)/sqrt(size(ICCmat3{clusternum},1))],'color',[0 0 0]);
            line(((clusternum-1)*3+[1 2]),0.6*[1 1],'color',[0 0 0]);
            shift=0;
            if p2(clusternum,2)<0.001
                if pval_or_stars==1,ptext='{\itp}<0.001';else,ptext='***';shift=0.082;end
            elseif p2(clusternum,2)<0.01
                if pval_or_stars==1,ptext='{\itp}<0.01';else,ptext='**';shift=0.082;end
            elseif p2(clusternum,2)<0.05
                if pval_or_stars==1,ptext='{\itp}<0.05';else,ptext='*';shift=0.082;end
            elseif p2(clusternum,2)<0.1
                if pval_or_stars==1,ptext='{\itp}<0.1';else,ptext='~';end
            else
                if pval_or_stars==1,ptext=['{\itp}=',num2str(round(100*p2(clusternum,2))/100)];else,ptext='n.s';end
            end
            text(((clusternum-1)*3+1.5),0.6-shift,ptext,'horizontalalignment','center','verticalalignment','bottom','fontsize',16);
        end
        if ybreak==1
            set(gca,'xtick',[1,2,4,5,7,8,10,11],'xticklabel',{'Real','Shuf','Real','Shuf','Real','Shuf','Real','Shuf','Real','Shuf'},'fontsize',16);
            xtickangle(gca,90);
            ylim([-15 -13]);set(gca,'ytick',[-15 -13]);
        else
            set(gca,'xtick',[],'fontsize',16);
            ylb=ylabel('ICC within-set');
            set(ylb,'position',[-1.525,-1.263,-1]);        
            ylim([-3 1]);set(gca,'ytick',[-2 -1 0 1]);
        end
        drawnow;
        if drawoption==1
            set(gca,'position',get(gca,'position').*[1 1 0.8 1]);
        end
    end

    for corrtype=[3 1 2]
        switch corrtype
            case 1
                if drawoption==1
                    subplot(2,12,22:24);
                else
                    subplot('position',[0.624    0.1100    0.1159    0.3412]);
                end
                ylab=ylabel('Non-cued objects');
            case 2
                if drawoption==1
                    subplot(2,12,19:21);
                else
                    subplot('position',[0.4815    0.1100    0.1159    0.3412]);
                end

                ylab=ylabel(['\Delta memory for objects ',char(8712),' cued sets (\DeltaESDF)\newline  Less forgetting <---------> More Forgetting\newline\newline                       Cued objects']);% % % $ %
                
            case 3
                if drawoption==1
                    continue;
                else
                    subplot('position',[0.8132    0.1100    0.1159    0.3412]);
                end
                ylab=ylabel(['Forgetting slopes for objects ',char(8713),' cued sets\newline  Less forgetting <---------> More Forgetting']);% % % $ %
        end
        if reg_or_zscore==2
            xlab=xlabel('ICC within-set (Z-score)');        
            set(ylab,'position',[-1.1 0 0]);        
            set(xlab,'position',[0 -1.6 0]);        
        elseif reg_or_zscore==1
            xlab=xlabel('ICC within-set');        
            set(ylab,'position',[-3.1 0 0]);        
            set(xlab,'position',[-1 -1.6 0]);        
        end
        hold all
        clusters2draw=[2,4];
        for clusternum=clusters2draw
            y=behav.subdata(subloc1,corrtype); 
            x=ZIcc{clusternum,2}(subloc2); 
            [x,k] = sort(x);
            y = y(k);
            surface_area_model = fitlm(x,y,'linear');
            [p,s]=polyfit(x,y,1);
            [yfit,dy]=polyconf(p,x,s,'predopt','curve');
            a= 25;
            h1= scatter(x,y,a,'o','LineWidth',0.5,'MarkerEdgeColor',clrs(clusternum,:),'MarkerFaceColor',clrs(clusternum,:));
            line(x,yfit,'color',clrs(clusternum,:),'LineWidth',0.5);
            line(x,yfit-dy,'color',clrs(clusternum,:),'linestyle','--');
            line(x,yfit+dy,'color',clrs(clusternum,:),'linestyle','--');
            line([0 0],[-1.5 1.5],'color','k');
            line(get(gca,'xlim'),[0 0],'color','k');
            drawnow;
            set(gca,'xtick',get(gca,'xlim'),'ytick',sort([0,get(gca,'ylim')]));
        end
        if reg_or_zscore==2
            set(gca,'ylim',[-1.5 1.5],'xlim',[-1 1],'ytick',[-1.5 1.5],'xtick',[-1 1]);
        elseif reg_or_zscore==1
            set(gca,'ylim',[-1.5 1.5],'xlim',[-3 1],'ytick',[-1.5 1.5],'xtick',[-3 1]);
        end
        if corrtype==3
            set(gca,'ylim',[-0.3 1.3],'ytick',[-0.3 1.3]);
            set(ylab,'position',[-3.863 0.5 0]);        
            set(xlab,'position',[-1 -0.4 0]);        
        end
        drawnow;
        if p_r(clusters2draw(1),2+corrtype)<0.001
            if pval_or_stars==1,ptext='{\itp}<0.001';else,ptext='***';end
        elseif p_r(clusters2draw(1),2+corrtype)<0.01
            if pval_or_stars==1,ptext='{\itp}<0.01';else,ptext='**';end
        elseif p_r(clusters2draw(1),2+corrtype)<0.05
            if pval_or_stars==1,ptext='{\itp}<0.05';else,ptext='*';end
        elseif p_r(clusters2draw(1),2+corrtype)<0.1
            if pval_or_stars==1,ptext='{\itp}<0.1';else,ptext='~';end
        else
            if pval_or_stars==1,ptext=['{\itp}=',num2str(round(100*p_r(clusters2draw(1),2+corrtype))/100)];else,ptext=', n.s';end
        end
        text(min(get(gca,'xlim')),min(get(gca,'ylim'))+diff(get(gca,'ylim'))*0.075,[' r = ',num2str(round(100*r(clusters2draw(1),2+corrtype))/100),ptext],'color',clrs(clusters2draw(1),:),'horizontalalignment','left','verticalalignment','bottom','fontsize',15);
        if p_r(clusters2draw(2),2+corrtype)<0.001
            if pval_or_stars==1,ptext=', {\itp}<0.001';else,ptext='***';end
        elseif p_r(clusters2draw(2),2+corrtype)<0.01
            if pval_or_stars==1,ptext=', {\itp}<0.01';else,ptext='**';end
        elseif p_r(clusters2draw(2),2+corrtype)<0.05
            if pval_or_stars==1,ptext=', {\itp}<0.05';else,ptext='*';end
        elseif p_r(clusters2draw(2),2+corrtype)<0.1
            if pval_or_stars==1,ptext=', {\itp}<0.1';else,ptext='~';end
        else
            if pval_or_stars==1,ptext=[', {\itp}=',num2str(round(100*p_r(clusters2draw(2),2+corrtype))/100)];else,ptext=', n.s';end
        end
        text(min(get(gca,'xlim')),min(get(gca,'ylim')),[' r = ',num2str(round(100*r(clusters2draw(2),2+corrtype))/100),ptext],'color',clrs(clusters2draw(2),:),'horizontalalignment','left','verticalalignment','bottom','fontsize',15);
        set(gca,'fontsize',15);
        drawnow;
        if drawoption==1
            set(gca,'position',get(gca,'position').*[1 1 0.8 1]+[0.05 0 0 0]);
        end
    end

catch err
    show_err(err,1);
    1
end

