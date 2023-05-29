`timescale 1ns / 1ps

module priorityEncoder(d, x, v);

input [3:0] d;
output reg [1:0] x;
output reg v;
integer k = 0;
always@(d)
 begin
 x = 2'bxx;
 v = 0;
 for(k=0;k<4;k=k+1)
  if (d[k])
   begin
    x = k;
    v = 1;
   end
 end

endmodule


