#!/usr/bin/perl -w

use strict;
use Getopt::Long;
use Cwd qw(abs_path);
use File::Basename qw(basename dirname);



########
#Interface
########


my $version="0.9";

#v0.2, add --type to accept different input types
#v0.3, add excel output
#v0.3a, add folder creation, solved failed tests issue
#v0.41, AWS
#v0.5, add --cate for directional changes
#v0.6, add zscore calculation
#v0.7, add barplot
#v0.8, redefined N (bk)
#v0.81, fixed barplot bug
#v0.9, add waterfall plot


my $usage="

gstools gs-fisher
version: $version
Usage: gstools gs-fisher [parameters]

Description: Perform fisher's exact test for select genes for a list of gene sets(gs)


Mandatory Parameters:
    --in|-i           Input file

    --type|-t         Input data type [list]
                          list, gene list in each column
                          matrix, significance matrix, e.g. from rnaseq-summary
                          de, DE gene diff results from rnaseq-de					  

    --cate|-c         Category of changes, both or updown [both]
                        this option works for --type matrix or --type de
                        use both for list including both up and down
                        use updown to generate lists for up and down separately

    --out|-o          Output folder
    --db|-d           Database file
    --backgroud|-b    Background file (optional)

    --anno|-a         Annotation file for gene list to symbols

    --qmethod         Multiple testing correction method [BH]
    --qcutoff         Corrected P cutoff [0.05]
    --top             No. of top terms to show in barplot [20]
	
    --verbose|-v      Verbose
	
";


unless (@ARGV) {
	print STDERR $usage;
	exit;
}

my $params=join(" ",@ARGV);
#then call different scripts



########
#Parameters
########
	
my $infile;
my $dbfile; #gene to gs annotation
my $annofile; #gene id to symbol
my $bkfile;
my $outfolder;

my $type="list";
my $cate="both"; #both for both up/down genes, updown will separate

my $pmethod="fisher";
my $qmethod="BH";
my $alternative="greater"; #greater, less
my $qcutoff=0.05;
my $topnum=20;
my $bkmodel="total"; #total, selected
my $sortby="pvalue"; #pvalue, depth, name

	
my $verbose=1;
my $runmode="none";
my $dev=0; #developmental version

GetOptions(
	"input|i=s" => \$infile,
	"db|d=s" => \$dbfile,
	"type|t=s" => \$type,
	#"pmethod=s"=>\$pmethod,
	"cate|c=s" => \$cate,
	"anno|a=s"=> \$annofile,
	"qmethod=s"=>\$qmethod,
	"qcutoff=f"=>\$qcutoff,
	"alternative=s"=>\$alternative,
	"background|b=s"=>\$bkfile,
	"top=s"=>\$topnum,
	"sort=s"=> \$sortby,
	"out|o=s"=>\$outfolder,
	"verbose"=>\$verbose,
	"dev" => \$dev,	
);



########
#Prerequisites
########

my $r="/usr/bin/R";
my $rscript="/usr/bin/Rscript";

#Application version


my $omictoolsfolder="/apps/omictools/";

#adding --dev switch for better development process
if($dev) {
	$omictoolsfolder="/home/jyin/Pipeline/omictools/";
#}
#else {
	#the tools called will be within the same folder of the script
	#$omictoolsfolder=get_parent_folder(abs_path(dirname($0)));
}

my $mergefiles="$omictoolsfolder/mergefiles/mergefiles_caller.pl";
my $text2excel="perl $omictoolsfolder/text2excel/text2excel.pl";


#gstools
my $gstoolsfolder="/apps/gstools/";

#adding --dev switch for better development process
if($dev) {
	$gstoolsfolder="/home/jyin/Pipeline/gstools/";
}

my $gs_barplot="$gstoolsfolder/plots/gs_barplot.R";
my $gs_waterfallplot="$gstoolsfolder/plots/gs_waterfallplot.R";

########
#Program running
########


print STDERR "\nomictools gs-fisher $version running ...\n\n" if $verbose;

#mkdir if the dir doesn't exist
#my $outfiledirname  = dirname($outfile);

#mkpath for recursive make dir
if(!-e $outfolder) {
	print STDERR "$outfolder not found. mkdir $outfolder\n";
	mkdir($outfolder); #doesn't do recursive
}

$outfolder=abs_path($outfolder);
my $outfoldername=basename($outfolder);

#the order of dirname and abs_path is important
my $outfile = "$outfolder/$outfoldername.xlsx";


my $logfile="$outfolder/gs-fisher_run.log";


#write log file
open(LOG, ">$logfile") || die "Error writing into $logfile. $!";

my $now=current_time();

print LOG "perl $0 $params\n\n";
print LOG "Start time: $now\n\n";
print LOG "Current version: $version\n\n";

print LOG "\n";


print LOG "\nomictools gs-fisher $version running ...\n\n";




########
#Process
########


my $filefolder="temp";

my $rand=randstring();


#read cate assignment
my %cates=map {$_,1} split(",",$cate);


my %selgenes;
my %gene_id;
my %gs2gene;


#####
#BK file
#####

#if bk file defined, use bk file. Otherwise, all the genes in the db.

my $blinenum=0;
my %bkgenes;
if(defined $bkfile && length($bkfile)>0) {
	open(IN,$bkfile) || die "ERROR:Can't read $bkfile.$!\n";
	
	while(<IN>) {
		tr/\r\n//d;
		unless($blinenum==0) {
			my @array=split/\t/;
			$bkgenes{$array[0]}++;
			#$gene_used{$array[0]}++;
		}
		$blinenum++;
	}
	close IN;
	
	print STDERR scalar(keys %bkgenes)," features found in the $bkfile file.\n" if $verbose;
	print LOG scalar(keys %bkgenes)," features found in the $bkfile file.\n";
}
else {
	print STDERR "No background file defined.\n" if $verbose;
	print LOG "No background file defined.\n";
}
	
#########
#GS db File
#########
my %gene_used;

my $alinenum=0;
open(ANNO,$dbfile) || die "ERROR:Can't read $dbfile.$!\n";
while(<ANNO>) {
	tr/\r\n\"\'//d;
	
	unless($alinenum==0) {
		my @array=split/\t/;
		
		#bk file defined, use intersection of bk and db
		if(defined $bkfile && length($bkfile)>0) {
			unless(defined $bkgenes{$array[0]}) {
				next;
			}
		}
		
		#genes in the annotation, by intersection with bk
		$gene_used{$array[0]}++;
		
		#add length>0
		if(defined $array[1] && length($array[1]) >0 && $array[1] ne " ") {
	
			foreach my $gs (split(";",$array[1])) {
				if($gs eq "-" || $gs eq " ") {
					next;
				}
				
				$gs2gene{$gs}{$array[0]}++;
			}
		}

	}
	
	$alinenum++;
}
close ANNO;

#bk genes should be the intersection of bk file and db file
#to implement that use 
%bkgenes=%gene_used;
print STDERR scalar(keys %bkgenes)," features used as the background.\n" if $verbose;
print LOG scalar(keys %bkgenes)," features used as the background.\n";

#previously implementation, if bk defined, use bk, regardless of whether it has annotation or not
#unless(defined $bkfile && length($bkfile)>0) {
#	%bkgenes=%gene_used; #use all genes in the anno as bk
#}



#########
#Annotation File
#########

#e.g. replace gene ID to symbol

my %gene2anno;
if(defined $annofile && length($annofile)>0) {
	my ($annnofilein,$genecol,$annocol)=split(",",$annofile);
	
	unless(defined $genecol && length($genecol)>0) {
		$genecol=1;
		$annocol=2;
	}
	
	my $clinenum=0;
	open(IN,$annnofilein) || die $!;
	while(<IN>) {
		tr/\r\n//d;
		unless ($clinenum==0) {
			my @array=split/\t/;
			$gene2anno{$array[$genecol-1]}=$array[$annocol-1];
		}
		$clinenum++;
	}
	close IN;
}



######
#Input
######

#support multiple columns of genes
my @comparison_names;
open(IN,$infile) || die "ERROR:Can't read $infile.$!\n";

#only matrix and de can do z-score calculation

if($type eq "list") {

	#
	my @contents=<IN>;
	my $title=shift @contents;
	$title=~tr/\r\n//d;
	@comparison_names=split(/\t/,$title);

	foreach (@contents) {
		tr/\r\n\"//d;
		my @array=split/\t/;
		my $num=-1;
		foreach my $id (@array) {
			$num++;
			
			if(defined $id && length($id)>0 && $id ne " ") {
				if(defined $bkgenes{$id}) {
					$selgenes{$comparison_names[$num]}{$id}=1;
				}
			}
		}
	}
	close IN;	
}
elsif($type eq "matrix") {
	my $ilinenum=0;
	#open(IN,$infile) || die "ERROR:Can't read $infile.$!\n";
	while(<IN>) {
		tr/\r\n//d;
		my @array=split/\t/;
		
		if($ilinenum==0) {
			@comparison_names=@array[1..$#array];
		}
		else {
			#only count genes in the bk
			if (defined $bkgenes{$array[0]}) {			
				for(my $num=1;$num<@array;$num++) {				
					if(defined $array[$num] && length($array[$num])>0 && $array[$num] ne "NA" && $array[$num] ne " " && $array[$num] ne "0") {
						if(defined $cates{"both"}) {
							if($array[$num]>0) {
								$selgenes{$comparison_names[$num-1]}{$array[0]}=1; #in order to make it work, it has to be -1/1
							}
							else {
								$selgenes{$comparison_names[$num-1]}{$array[0]}=-1; #in order to make it work, it has to be -1/1
							}
						}
						
						#probably no need for this anymmore
						if(defined $cates{"updown"}) {
							if($array[$num]==1) {
								$selgenes{$comparison_names[$num-1]."-up"}{$array[0]}=1;
							}
							elsif($array[$num]==-1) {
								$selgenes{$comparison_names[$num-1]."-down"}{$array[0]}=-1;
							}
						}
					}
				}
			}
		}
		$ilinenum++;
	}
	close IN;
}
elsif($type eq "de") {
	#rnaseq-de result
	my $ilinenum=0;
	while(<IN>) {
		tr/\r\n//d;
		my @array=split/\t/;
		
		if($ilinenum==0) {
			#guess name			
			if($array[1]=~/log2 fold change (MLE): (.+)/) {
				my $compname=$1;
				$compname=~s/\W/_/g;
				push @comparison_names,$compname;
			}
			elsif($array[1]=~/log2 fold change (.+)/) {
				my $compname=$1;
				$compname=~s/\W/_/g;
				push @comparison_names,$compname;
			}
		}
		else {
			my $num=5;
			
			if (defined $bkgenes{$array[0]}) {
				
				#use both for now #non-0
				if(defined $cates{"both"}) {
					if(defined $array[$num] && length($array[$num])>0 && $array[$num] ne "NA" && $array[$num] ne " " && $array[$num] ne "0") {
						if($array[$num]>0) {
							$selgenes{$comparison_names[0]}{$array[0]}=1;
						}
						else {
							$selgenes{$comparison_names[0]}{$array[0]}=-1;
						}
					}
				}
				
				#probably no need for this anymmore
				if(defined $cates{"updown"}) {
					if($array[$num]==1) {
						$selgenes{$comparison_names[$num-1]."-up"}{$array[0]}=1;
					}
					elsif($array[$num]==-1) {
						$selgenes{$comparison_names[$num-1]."-down"}{$array[0]}=-1;
					}
				}
			}				
		}
		$ilinenum++;
	}
	close IN;
}


my @final_comparison_names=sort keys %selgenes;

print STDERR "Sample list(s) ",join(",",@final_comparison_names)," are identified from $infile.\n\n" if $verbose;
print LOG "Sample list(s) ",join(",",@final_comparison_names)," are identified from $infile.\n\n";

foreach my $compname (@final_comparison_names) {
	print STDERR scalar(keys %{$selgenes{$compname}})," genes are identified from $compname.\n" if $verbose; 
	print LOG scalar(keys %{$selgenes{$compname}})," genes are identified from $compname.\n";
}


#######
#Calculation
#######

my $all_num=keys %bkgenes;

#my $outfolder=abs_path(dirname($outfile));

if(!-e "$outfolder/$filefolder") {
	mkdir("$outfolder/$filefolder");
}

my $tempfile="$outfolder/$filefolder/gs-fisher_tem_$rand.txt";

#if(@comparison_names<=1) {
#	my $exp_id=$comparison_names[0];
#}
#else {
	#should only use this after DE_sigs

my @outfiles;
my @outfilenames;
	
foreach my $exp_id (@final_comparison_names) {
	open(TMP,">$tempfile") || die $!;
	my %gs2gene_sel; 
	my %changed_gene; #n
	
	foreach my $gs (sort keys %gs2gene) {
		foreach my $id (sort keys %{$selgenes{$exp_id}}) {

			#next unless defined $bkgenes{$id};
			
			$changed_gene{$id}++;
			
			if(defined $gs2gene{$gs}{$id}) {
				$gs2gene_sel{$gs}{$id}++;
			}
		}
	}
	
	my $total_num=keys %changed_gene; #n
	
	print TMP "Total\t$all_num\t$total_num\t-\t-\n"; #N, n,genes,z-score
	
	my $k_num;
	
	foreach my $gs (sort keys %gs2gene) {
		if(defined $gs2gene_sel{$gs}) {
			my $num2=keys %{$gs2gene_sel{$gs}}; #m
			my $num1=keys %{$gs2gene{$gs}}; #M
			
			my %annos;
			
			my $ztotal;
			my $zscore;
			
			if(defined $annofile && length($annofile)>0) {
				foreach my $gene (sort keys %{$gs2gene_sel{$gs}}) {
					if(defined $gene2anno{$gene}) {
						$annos{$gene2anno{$gene}}++;
					}
					else {
						$annos{$gene}++;
					}								
				}
			}
			else {
				%annos=%{$gs2gene_sel{$gs}};
			}
			
			#z-score calculation
			foreach my $gene (sort keys %{$gs2gene_sel{$gs}}) {
				if(defined $selgenes{$exp_id}{$gene}) {
					$ztotal+=$selgenes{$exp_id}{$gene};
				}
				else {
					print STDERR "WARNING:$gene not defined in $exp_id.\n";
				}								
			}
			
			$zscore=$ztotal/sqrt($num2); #total/sqrt(m)
									
			#M,m,genes,z-score,
			print TMP $gs,"\t",$num1,"\t",$num2,"\t",join(";",sort keys %annos),"\t",$zscore,"\n"; #M, m
		
		}
	}
	close TMP;
	
	my $newid=$exp_id;
	$newid=~s/\W/_/g;
	
	#v0.3 file name
	#my $file_out=$outfile;
	#$file_out=~s/\.\w+$/_$newid\.txt/;
	
	#new file name for better merging purpose
	
	my $file_out="$outfolder/$newid\_gs-fisher.txt";
	
	#substr($outfile,0,length($outfile)-4)."_$newid.txt";
	
	#remember file names

	
	print STDERR "\nResult File,$file_out\t" if $verbose;
	print LOG "\nResult File,$file_out\t";

	
	path_r($tempfile,$file_out);
	
	print STDERR "\ndone!\n" if $verbose;
	print LOG "\ndone!\n";
	
	if(-e $file_out) {
		#double check the files, because sometimes tests can fail
		push @outfiles,$file_out;
		push @outfilenames,$newid;	
	}
	else {
		print STDERR "WARNING: $exp_id failed test. Skip $file_out.\n\n";
		print LOG "WARNING: $exp_id failed test. Skip $file_out.\n\n";
	}
	
	##unlink($tempfile); #masked for test
	
	#plto bar plot. Top 20 and All sigs
	system("$rscript $gs_barplot -i $file_out -o $outfolder/$newid\_gs-fisher_top$topnum\_barplot.png --top $topnum --siglevel $qcutoff");
	print LOG "$rscript $gs_barplot -i $file_out -o $outfolder/$newid\_gs-fisher_top$topnum\_barplot.png --top $topnum --siglevel $qcutoff\n";

	system("$rscript $gs_barplot -i $file_out -o $outfolder/$newid\_gs-fisher_all_barplot.png --top all --siglevel $qcutoff");
	print LOG "$rscript $gs_barplot -i $file_out -o $outfolder/$newid\_gs-fisher_all_barplot.png --top all --siglevel $qcutoff\n";
	
	#plto bar plot. Top 20 and All sigs
	system("$rscript $gs_waterfallplot -i $file_out -o $outfolder/$newid\_gs-fisher_top$topnum\_waterfallplot.png --top $topnum --siglevel $qcutoff");
	
	print LOG "$rscript $gs_waterfallplot -i $file_out -o $outfolder/$newid\_gs-fisher_top$topnum\_waterfallplot.png --top $topnum --siglevel $qcutoff\n";

	system("$rscript $gs_waterfallplot -i $file_out -o $outfolder/$newid\_gs-fisher_all_waterfallplot.png --top all --siglevel $qcutoff");
	
	print LOG "$rscript $gs_waterfallplot -i $file_out -o $outfolder/$newid\_gs-fisher_all_waterfallplot.png --top all --siglevel $qcutoff\n";
}


#Merge files into excel file
my $exceloutfile=$outfile;
#$exceloutfile=~s/\.\w+$/\.xlsx/;

print STDERR "Converting outputs to excel file:$exceloutfile.\n\n";
print LOG "Converting outputs to excel file:$exceloutfile.\n\n";

system("$text2excel -i ".join(",",@outfiles)." -n ".join(",",@outfilenames)." -o $exceloutfile --theme theme2");
print LOG "$text2excel -i ".join(",",@outfiles)." -n ".join(",",@outfilenames)." -o $exceloutfile --theme theme2\n";


close LOG;


			
###################################
#R scripts here
###################################
sub max {
	my @nums=@_;
	my $max=$nums[0];
	foreach (@nums) {
		if($_>$max) {
			$max=$_;
		}
	}
	return $max;
}

sub path_r {
	my ($file_in,$file_out)=@_;
	open(RPROG,"|$r --no-restore --no-save --slave") || die $!;

	select RPROG;
	print <<"CODE";
	
###This script reads the pathway list from microarray result.
###The P value is calculated by Hypergeometric distribution, a package in R.
###The Q value is calculated by a R package integrated below, which was from CRAN.
###By default, FDR is set to 0.05.

path_stat <- function(file_in,file_out,pmethod=c("hyper","fisher"),qmethod="BH",Q=0.05,alternative="greater",sortby="pvalue") {
	table<-read.table(file_in,header=FALSE,sep="\t") #row.names=1
	m<-table[1,3]	#n
	n<-table[1,2]-table[1,3]	#N-n
	k<-table[,2]	#M
	x<-table[,3]	#m
	num<-2;
	r_enrich<-c()
	
	#fisher
	if(pmethod=="fisher") {
		p_hyper<-c()
		while(num <= length(x)) {
			data<-matrix(c(x[num],m-x[num],k[num]-x[num],n-k[num]+x[num]),nr=2)
			p_fisher<-fisher.test(data,alternative=alternative)
			
			p_hyper[num]<-p_fisher\$p.value

			if(is.infinite(p_fisher\$estimate)) {
				#an aritificial OR to replace inf
				r_enrich[num]<-(x[num]+0.1)/(k[num]-x[num]+0.1)/(m-x[num]+0.1)*(n-k[num]+x[num]+0.1)
			}
			else {
				r_enrich[num]<-p_fisher\$estimate
			}
			
			num<-num+1
		}
		realp<-p_hyper[2:length(p_hyper)]
		realr<-r_enrich[2:length(r_enrich)]
	}

	#Enrichment factor
	#num<-2
	#r_enrich<-c()
	#while(num <= length(x)) {
	#	r_enrich[num]<- x[num]*(n+m)/(k[num]*m)
	#	num<-num+1
	#}
	#realr<-r_enrich[2:length(r_enrich)]
	
	
    if(qmethod == "BH") {
		qvalue<-p.adjust(realp,method="BH");
		sig<-qvalue
		sig[qvalue<Q]="TRUE";
		sig[qvalue>=Q]="FALSE";
		#qvalue<-c("-",qvalue)
		#sig<-c('-',sig);
	} else {
		qvalue<-p.adjust(realp,method=qmethod);
		sig<-qvalue
		sig[qvalue<Q]="TRUE";
		sig[qvalue>=Q]="FALSE";
		#qvalue<-c("-",qvalue)
		#sig<-c('-',sig);	
	}
	
    #pvalue<-c('-',realp)
	#rvalue<-c('-',realr)
	
	totalline<-c("Total",table[1,2],table[1,3],"-","-","-","-","-","-")
	#content<-data.frame(cbind(table,pvalue,rvalue,qvalue,sig))
	content<-cbind(as.matrix(table[2:nrow(table),]),realp,realr,qvalue,sig)
	
	#sort function
	if(nrow(content)>1) {
		if(sortby=="pvalue") {
			content<-content[order(realp),]
		} else if(sortby=="name"){
			content<-content[order(content[,1]),]
		}
	}
	
	result<-rbind(totalline,as.matrix(content))
	
	colnames(result)<-c("Gene set",'No. of features in the gene set','No. of selected features in the gene set','Feature ID','Z-score',paste(c('P value',pmethod,alternative),collapse=" "),'Odds Ratio',paste(c(qmethod,'Corrected P value'),collapse=" "),paste(c("Significance By",qmethod,'Corrected P value', Q),collapse=" "))

	write.table(result,file=file_out,sep="\\t",row.names=FALSE,quote=F)
	#cat(log)
	#pathout<-"path_stat.log"
	#write(log,file=pathout,append = TRUE)
}

path_stat("$file_in","$file_out",Q=$qcutoff,qmethod="$qmethod",pmethod="$pmethod",alternative="$alternative",sortby="$sortby");

#warnings()

q()
CODE
close RPROG;

}

sub randstring {
	my @chars=("A".."Z");
	my $string;
	
	$string.=$chars[rand(@chars)] for 1..6;
	
	return $string;
}


sub current_time {
	my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
	my $now = sprintf("%04d-%02d-%02d %02d:%02d:%02d", $year+1900, $mon+1, $mday, $hour, $min, $sec);
	return $now;
}


sub getsysoutput {
	my $command=shift @_;
	my $output=`$command`;
	$output=~tr/\r\n//d;
	return $output;
}


sub build_timestamp {
	my ($now,$opt)=@_;
	
	if($opt eq "long") {
		$now=~tr/ /_/;
		$now=~tr/://d;
	}
	else {
		$now=substr($now,0,10);
	}
	
	return $now;
}


sub find_program {
	my $fullprogram=shift @_;
	
	#use defined program as default, otherwise search for this program in PATH
	
	my $program;
	if($fullprogram=~/([^\/]+)$/) {
		$program=$1;
	}
	
	if(-e $fullprogram) {
		return $fullprogram;
	}
	else {
		my $sysout=`$program`;
		if($sysout) {
			my $location=`which $program`;
			return $location;
		}
		else {
			print STDERR "ERROR:$fullprogram or $program not found in your system.\n\n";
			exit;
		}
	}
}


sub get_parent_folder {
	my $dir=shift @_;
	
	if($dir=~/^(.+\/)[^\/]+\/?/) {
		return $1;
	}
}

