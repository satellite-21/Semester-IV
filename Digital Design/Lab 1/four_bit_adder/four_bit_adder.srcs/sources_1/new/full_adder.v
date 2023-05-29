`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 13.01.2022 01:50:47
// Design Name: 
// Module Name: full_adderFA
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module full_adder(co, s, a, b, ci);
input a, b, ci;
output co, s;
wire c1, s1, c2;

half_adder H1(c1, s1, a, b);
half_adder H2(c2, s, s1, ci);
or (co, c2, c1);
endmodule

module half_adder(c, s, a, b);
input a,b;
output s, c;
assign s = a^b;
assign c = a&b;

endmodule
