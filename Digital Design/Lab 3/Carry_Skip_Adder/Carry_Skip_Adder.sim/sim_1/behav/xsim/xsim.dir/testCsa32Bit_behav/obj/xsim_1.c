/**********************************************************************/
/*   ____  ____                                                       */
/*  /   /\/   /                                                       */
/* /___/  \  /                                                        */
/* \   \   \/                                                         */
/*  \   \        Copyright (c) 2003-2013 Xilinx, Inc.                 */
/*  /   /        All Right Reserved.                                  */
/* /---/   /\                                                         */
/* \   \  /  \                                                        */
/*  \___\/\___\                                                       */
/**********************************************************************/


#include "iki.h"
#include <string.h>
#include <math.h>
#ifdef __GNUC__
#include <stdlib.h>
#else
#include <malloc.h>
#define alloca _alloca
#endif
/**********************************************************************/
/*   ____  ____                                                       */
/*  /   /\/   /                                                       */
/* /___/  \  /                                                        */
/* \   \   \/                                                         */
/*  \   \        Copyright (c) 2003-2013 Xilinx, Inc.                 */
/*  /   /        All Right Reserved.                                  */
/* /---/   /\                                                         */
/* \   \  /  \                                                        */
/*  \___\/\___\                                                       */
/**********************************************************************/


#include "iki.h"
#include <string.h>
#include <math.h>
#ifdef __GNUC__
#include <stdlib.h>
#else
#include <malloc.h>
#define alloca _alloca
#endif
typedef void (*funcp)(char *, char *);
extern int main(int, char**);
extern void execute_107(char*, char *);
extern void execute_108(char*, char *);
extern void execute_314(char*, char *);
extern void execute_315(char*, char *);
extern void vlog_const_rhs_process_execute_0_fast_no_reg_no_agg(char*, char*, char*);
extern void vlog_simple_process_execute_0_fast_no_reg_no_agg(char*, char*, char*);
extern void execute_135(char*, char *);
extern void execute_136(char*, char *);
extern void execute_137(char*, char *);
extern void execute_138(char*, char *);
extern void execute_119(char*, char *);
extern void execute_115(char*, char *);
extern void execute_116(char*, char *);
extern void execute_110(char*, char *);
extern void execute_111(char*, char *);
extern void execute_112(char*, char *);
extern void execute_316(char*, char *);
extern void execute_317(char*, char *);
extern void execute_318(char*, char *);
extern void execute_319(char*, char *);
extern void execute_320(char*, char *);
extern void vlog_transfunc_eventcallback(char*, char*, unsigned, unsigned, unsigned, char *);
funcp funcTab[22] = {(funcp)execute_107, (funcp)execute_108, (funcp)execute_314, (funcp)execute_315, (funcp)vlog_const_rhs_process_execute_0_fast_no_reg_no_agg, (funcp)vlog_simple_process_execute_0_fast_no_reg_no_agg, (funcp)execute_135, (funcp)execute_136, (funcp)execute_137, (funcp)execute_138, (funcp)execute_119, (funcp)execute_115, (funcp)execute_116, (funcp)execute_110, (funcp)execute_111, (funcp)execute_112, (funcp)execute_316, (funcp)execute_317, (funcp)execute_318, (funcp)execute_319, (funcp)execute_320, (funcp)vlog_transfunc_eventcallback};
const int NumRelocateId= 22;

void relocate(char *dp)
{
	iki_relocate(dp, "xsim.dir/testCsa32Bit_behav/xsim.reloc",  (void **)funcTab, 22);

	/*Populate the transaction function pointer field in the whole net structure */
}

void sensitize(char *dp)
{
	iki_sensitize(dp, "xsim.dir/testCsa32Bit_behav/xsim.reloc");
}

void simulate(char *dp)
{
		iki_schedule_processes_at_time_zero(dp, "xsim.dir/testCsa32Bit_behav/xsim.reloc");
	// Initialize Verilog nets in mixed simulation, for the cases when the value at time 0 should be propagated from the mixed language Vhdl net
	iki_execute_processes();

	// Schedule resolution functions for the multiply driven Verilog nets that have strength
	// Schedule transaction functions for the singly driven Verilog nets that have strength

}
#include "iki_bridge.h"
void relocate(char *);

void sensitize(char *);

void simulate(char *);

extern SYSTEMCLIB_IMP_DLLSPEC void local_register_implicit_channel(int, char*);
extern void implicit_HDL_SCinstatiate();

extern SYSTEMCLIB_IMP_DLLSPEC int xsim_argc_copy ;
extern SYSTEMCLIB_IMP_DLLSPEC char** xsim_argv_copy ;

int main(int argc, char **argv)
{
    iki_heap_initialize("ms", "isimmm", 0, 2147483648) ;
    iki_set_sv_type_file_path_name("xsim.dir/testCsa32Bit_behav/xsim.svtype");
    iki_set_crvs_dump_file_path_name("xsim.dir/testCsa32Bit_behav/xsim.crvsdump");
    void* design_handle = iki_create_design("xsim.dir/testCsa32Bit_behav/xsim.mem", (void *)relocate, (void *)sensitize, (void *)simulate, 0, isimBridge_getWdbWriter(), 0, argc, argv);
     iki_set_rc_trial_count(100);
    (void) design_handle;
    return iki_simulate_design();
}
