accuracy = 1e-06
regression_accuracy = 1e-08
name = 'KRR'
regression_labels = [-1, -1, -1, -1, 1, -1, -1, 1, 1, 1, 1]
data_train = [0.1374155, 0.0654813818, 0.871542014, 0.730734902, 0.317499288, 0.504578003, 0.135744184, 0.707160294, 0.0703305921, 0.757033057, 0.54683651;0.27885053, 0.862574354, 0.98607187, 0.254349101, 0.80624615, 0.750166316, 0.747190121, 0.957235611, 0.486617092, 0.674168041, 0.823877364;0.479536445, 0.481431203, 0.318654754, 0.780825713, 0.68619578, 0.4511399, 0.419336657, 0.0725896432, 0.00650114683, 0.528893967, 0.417492852;0.79603676, 0.542654056, 0.936706043, 0.196140799, 0.99015034, 0.297919194, 0.642423974, 0.501853714, 0.359576401, 0.139309183, 0.134834151;0.711505827, 0.377641383, 0.961297974, 0.325648487, 0.983523722, 0.205423257, 0.102814386, 0.312569734, 0.885584789, 0.521851944, 0.135650106;0.20249905, 0.311260662, 0.590166738, 0.33237894, 0.747054082, 0.616293556, 0.873823737, 0.500961826, 0.331176414, 0.996271568, 0.355633322;0.42860264, 0.809127225, 0.0886362339, 0.386490106, 0.0857100934, 0.139305867, 0.0046408255, 0.00954429258, 0.38028789, 0.809163805, 0.587004361;0.634914044, 0.519249238, 0.819793144, 0.880378725, 0.849740728, 0.812132106, 0.182713074, 0.593876225, 0.582163196, 0.527944004, 0.480812789;0.0891704772, 0.102163234, 0.90228157, 0.630186439, 0.262317214, 0.279300389, 0.610194832, 0.406934975, 0.441324171, 0.0606499905, 0.138739187;0.393350427, 0.69968642, 0.0480253876, 0.334557793, 0.695071952, 0.79205079, 0.190220704, 0.731487132, 0.955411694, 0.90681215, 0.789376077;0.712326686, 0.235138308, 0.633903227, 0.643273994, 0.406821332, 0.493015462, 0.694028488, 0.952101536, 0.295306208, 0.246363152, 0.601806892]
feature_class = 'simple'
regression_num_threads = 16
data_test = [0.130799111, 0.58253283, 0.0888514851, 0.49058388, 0.620414375, 0.0701096393, 0.206953152, 0.335374285, 0.368598695, 0.676104643, 0.529021341, 0.38210847, 0.450676254, 0.0129766707, 0.770525279, 0.0410046887, 0.829317493;0.540996521, 0.527819191, 0.704578628, 0.859559533, 0.942646146, 0.543369887, 0.150658232, 0.878630926, 0.0406720184, 0.807337171, 0.680421626, 0.904727607, 0.89746315, 0.0531540854, 0.609446407, 0.124671455, 0.0784686117;0.448277146, 0.0975975398, 0.727902383, 0.968940711, 0.569582958, 0.23516887, 0.843206392, 0.853103375, 0.732785972, 0.615132185, 0.867792455, 0.163291682, 0.931600126, 0.947203149, 0.871992374, 0.250662035, 0.858178386;0.497638212, 0.594930237, 0.840600623, 0.0295828445, 0.362942592, 0.815693377, 0.813954784, 0.848687927, 0.1692159, 0.705886618, 0.146503341, 0.785098009, 0.245126004, 0.737817728, 0.455001619, 0.986263465, 0.579893831;0.632579364, 0.368163514, 0.734520158, 0.494885641, 0.0180666351, 0.608581598, 0.83045959, 0.23680177, 0.221063755, 0.608886183, 0.817979276, 0.249620338, 0.932262651, 0.927387357, 0.200536672, 0.263784412, 0.700598811;0.951733109, 0.709882689, 0.44490005, 0.169001346, 0.461009849, 0.606767792, 0.330951919, 0.510257821, 0.130214128, 0.452188105, 0.0907758189, 0.0490887499, 0.857239916, 0.205081584, 0.575155329, 0.148982758, 0.262527571;0.36133942, 0.143676436, 0.045962297, 0.877159802, 0.0118328359, 0.0613209828, 0.805072161, 0.282008337, 0.722820301, 0.146225749, 0.237281749, 0.572776517, 0.988635838, 0.0734444702, 0.400554598, 0.04308097, 0.199226443;0.907471027, 0.0964601038, 0.856855921, 0.177803238, 0.861784847, 0.538021963, 0.878206674, 0.151703645, 0.762036551, 0.680698596, 0.542933387, 0.810587344, 0.648276024, 0.877046925, 0.381062147, 0.589923178, 0.870600843;0.214814064, 0.485741357, 0.960741822, 0.473601801, 0.252829445, 0.351076239, 0.885084119, 0.954883039, 0.950107643, 0.698942392, 0.509601889, 0.182688935, 0.123221012, 0.0597666076, 0.0177149095, 0.878116722, 0.809086316;0.567821381, 0.998723255, 0.953052072, 0.655108971, 0.464574567, 0.152170716, 0.515593805, 0.157743673, 0.398451304, 0.942648108, 0.341978989, 0.152787274, 0.582396604, 0.602364174, 0.193804242, 0.196586588, 0.989549791;0.243281073, 0.838192035, 0.470323826, 0.434989239, 0.294962416, 0.773825797, 0.880660878, 0.261127462, 0.0950966693, 0.136919553, 0.792803535, 0.435826911, 0.131914658, 0.340288206, 0.514235483, 0.852051684, 0.641743254]
data_type = 'double'
kernel_name = 'Gaussian'
regression_type = 'kernelmachine'
regression_tau = 1e-05
data_class = 'rand'
kernel_arg0_width = 1.5
regression_classified = [-0.681962784, 1.1222067, 0.0718337893, 0.540835459, -1.60665337, -0.772937945, -0.526927037, -1.01573466, -0.709138037, -0.100683158, -0.17364937, -1.06642117, 0.305093, -0.432059293, -0.487158295, -0.878494515, -0.204658793]
feature_type = 'Real'
