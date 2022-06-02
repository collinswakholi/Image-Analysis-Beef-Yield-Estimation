function data_out = incl_external(data_in, Inliers)

Age = cell2mat(Inliers.Age);
Sex = cell2mat(Inliers.Sex);
Weight = cell2mat(Inliers.HCW);

Sex1 = zeros(length(Sex),1);
Sex1(Sex=='F') = 1;
Sex1(Sex=='M') = 0.5;
Sex1(Sex=='C') = 0;

Weight = Weight/600;
Age = Age/250;
Sex = Sex1;

data_out = [data_in,Age,Sex,Weight];