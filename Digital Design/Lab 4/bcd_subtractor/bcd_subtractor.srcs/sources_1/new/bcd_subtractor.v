`timescale 1ns / 1ps
module tens_complement(x, t);
input [3:0] x;
output [3:0]t;
assign t[0] = x[0];
assign t[1] = (x[1]&x[0]) | (x[3]&!x[0]) | (x[2]&!x[1]&!x[0]);
assign t[2] = (x[2]&!x[1]) | (x[2]&!x[0]) | (!x[2]&!x[1]&!x[0]);
assign t[3] = (!x[2]&x[1]&!x[0]) | (!x[3]&!x[2]&!x[1]&x[0]);

endmodule
module bcd_adder(a,b,carry_in,sum,carry);
    input [3:0] a,b;
    input carry_in;
    output [3:0] sum;
    output carry;
    reg [4:0] sum_temp;
    reg [3:0] sum;
    reg carry;  
    always @(a,b,carry_in)
    begin
        sum_temp = a+b+carry_in;
        if(sum_temp > 9)    begin
            sum_temp = sum_temp+6;
            carry = 1;
            sum = sum_temp[3:0];    end
        else    begin
            carry = 0;
            sum = sum_temp[3:0];
        end
    end     

endmodule
module bcd_subtractor(a, b,c, s);
input [3:0] a;
input [3:0] b;
output [3:0] c;
output s;
    wire [3:0] tenscomp, tsum, tsum2;
    wire ct;
    
    tens_complement t(b, tenscomp);
    bcd_adder A(a, tenscomp, 1'b0, tsum, ct);
    assign s = !ct;
    
    tens_complement t2(tsum, tsum2);
    assign c = s?tsum2:tsum;
    
endmodule

