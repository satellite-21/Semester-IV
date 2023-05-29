`timescale 1ns / 1ps
module serial_adder(sum, cout, a, b, clk);
input [7:0] a, b;
input clk;
wire[7:0] x, z;
output [7:0] sum;
output cout;
wire s, cin;

fa fa1(s, cout, x[0], z[0], cin);
dff d1(cin, cout, clk);
sipo a1(sum, s, clk);
shift r1(x, a, clk);
shift r2(z, b, clk);
endmodule

module shift(y, d, clk);
input [7:0] d;
input clk;
output [7:0] y;
reg [7:0] y;

initial
    begin
    assign y = d;
    end
always @(posedge clk)
    begin
    assign y = y>>1;
    end
endmodule

module sipo(y, s, clk);
    input s;
    input clk;
    output [7:0] y;
    reg [7:0] y;
    always @(posedge clk)
    begin
    assign y = {s, y[7:1]};
    end
endmodule


module fa(s, cout, a, b, cin);
    input a, b, cin;
    output s, cout;
    assign {cout, s} = a+b+cin;
endmodule

module dff(q, d, clk);
input d, clk;
output q;
reg q;

initial 
begin
    q = 1'b0;
end

always @(posedge clk)
    begin
    q = d;
    end
    
endmodule