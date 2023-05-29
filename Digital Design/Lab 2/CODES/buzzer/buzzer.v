`timescale 1ns / 1ps
module buzzer(a1, a2, b1, b2, control);
input a1, a2, control;
output b1, b2;
wire c1, c2;
assign c1 = !(c2&a2) & control;
assign c2 = !(c1&a1) & control;

bufif1 (b1, a1, c1);
bufif1(b2, a2, c2);


endmodule
