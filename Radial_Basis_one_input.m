%% Radial Basis Function Neural Network for a 1-4-1 
% Calculate weights for the output layer

x = [0.0500; 0.2000; 0.2500; 0.3000; 0.4000; 0.4300; 0.4800; 0.6000; 0.7000; 0.8000; 0.9000; 0.95000];
z = [0.0863; 0.2662; 0.2362; 0.1687; 0.1260; 0.1756; 0.3290; 0.6694; 0.4573; 0.3320; 0.4063; 0.3535];
c = [];
% sigma = 0.1;
% c1 = 0.2;
% c2 = 0.6;
% c3 = 0.9;

c1 = (x(1,:)+x(2,:)+x(3,:))/3;
c2 = (x(4,:)+x(5,:))/2;
c3 = (x(6,:)+x(7,:)+x(8,:)+x(9,:))/4;
c4 = (x(10,:)+x(11,:)+x(12,:))/3;
rowav = (sqrt((c1-c2)^2)+sqrt((c1-c3)^2)+sqrt((c1-c4)^2)+sqrt((c2-c3)^2)+sqrt((c2-c4)^2)+sqrt((c3-c4)^2))/6;
sigma = 2*rowav;
for i = 1:size(x)
    if z(i,:) >= x(i,:)
      c = [c;1];
    else
      c = [c;2];
    end
end
Y = [];
for i = 1:size(x)
    d1 = sqrt((x(i,:)-c1)^2);
    d2 = sqrt((x(i,:)-c2)^2);
    d3 = sqrt((x(i,:)-c3)^2);
    d4 = sqrt((x(i,:)-c4)^2);
    Y1 = exp(-(d1^2/(2*sigma^2)));
    Y2 = exp(-(d2^2/(2*sigma^2)));
    Y3 = exp(-(d3^2/(2*sigma^2)));
    Y4 = exp(-(d4^2/(2*sigma^2)));
    Y = [Y;Y1 Y2 Y3 Y4 1];
end
W = (inv(Y'*Y)*Y')*z
% plot(x,z);hold on;
% plot(x,c)
