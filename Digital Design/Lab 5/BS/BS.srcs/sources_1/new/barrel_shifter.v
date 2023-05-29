`timescale 1ns / 1ps

module barrel_shifter(w,s,y);
input [3:0] w;
input [1:0] s;
output [3:0] y;

mux m1(w[3],w[0],w[1],w[2],y[3],s);
mux m2(w[2],w[3],w[0],w[1],y[2],s);
mux m3(w[1],w[2],w[3],w[0],y[1],s);
mux m4(w[0],w[1],w[2],w[3],y[0],s);


endmodule
