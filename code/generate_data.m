define_constants;
load_mask = [0;0;0;1;1;1;1;1;1];
for i = 1:1
    if mod(i,100) == 0
        i
    end
    mpc = loadcase('case9');
    ppert = (rand(9,1))*100;
    qpert = (rand(9,1) - 0.5)*100;
    %mpc.bus(:,PD) = times(load_mask, ppert);
    %mpc.bus(:,QD) = times(load_mask, qpert);
    mpopt = mpoption('verbose', 3, 'out.all', 1);
    results = runopf(mpc, mpopt);
    %result_file = sprintf('/Users/neelguha/Dropbox/NeelResearch/opf/data/result_%d', i);
    %mpc_file = sprintf('/Users/neelguha/Dropbox/NeelResearch/opf/data/mpc_%d', i);
    save(strcat(result_file), 'results');
    save(strcat(mpc_file), 'mpc');
end