properties([
    pipelineTriggers([githubPush()])
])

pipeline {
    agent any
    stages {
        stage('Checkout PR') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '${sha1}']],
                    extensions: [
                        [$class: 'CloneOption', depth: 1, shallow: true]
                    ],
                    userRemoteConfigs: [[
                        url: 'https://github.com/Camille0512/NLA.git',
                        credentialsId: 'github-token',
                        refspec: '+refs/pull/*:refs/remotes/origin/pr/*'
                    ]]
                ])
                sh 'echo "Finish PR checkout"'
            }
        }

        stage('Build & Test') {
            steps {
                sh 'pytest lu_decomposition_test.py'
                junit '**/JenkinsLogs/surefire-reports/*.xml'
                sh 'echo "Finish Build & Test"'
            }
        }
    }
}

post {
    always {
        cleanWs()  // Clean workspace after build
    }
}